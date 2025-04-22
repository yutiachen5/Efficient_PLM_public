from __future__ import print_function,division

import numpy as np
import sys
import os
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import average_precision_score as average_precision

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import PackedSequence
import torch.utils.data

from prose.utils import pack_sequences, unpack_sequences
from prose.utils import collate_paired_sequences, collate_lists, collate_seq2seq
from prose.utils import infinite_iterator, AllPairsDataset, MultinomialResample
from prose.utils import LargeWeightedRandomSampler
from prose.datasets import SCOPeDataset, SCOPePairsDataset, ContactMapDataset
from prose.datasets import FastaDataset, ClozeDataset

from prose.models.multitask import ProSEMT, OrdinalRegression, BilinearContactMap, L1, L2
from prose.models.lstm import SkipLSTM

def cmap_grad(model, x, y, use_cuda, weight=1.0):
    b = len(x)
    #if use_cuda:
    #    x = [x_.cuda() for x_ in x]
    x,order = pack_sequences(x)
    if use_cuda:
        x = x.cuda()

    z = model.transform(x) # embed the sequences

    # backprop each sequence individually for memory efficiency
    z_detach = z.data.detach()
    z_detach.requires_grad = True
    z_detach = PackedSequence(z_detach, z.batch_sizes)

    z_unpack = unpack_sequences(z_detach, order)

    # calculate loss for each sequence and backprop
    weight = weight/b

    loss = 0 # loss over minibatch
    tp = 0 # true positives over minibatch
    gp = 0 # number of ground truth positives in minibatch
    pp = 0 # number of predicted positives in minibatch
    total = 0 # total number of residue pairs

    for i in range(b):
        zi = z_unpack[i]
        logits = model.predict(zi.unsqueeze(0)).view(-1) # flattened predicted contacts
        yi = y[i].contiguous().view(-1) # flattened target contacts

        if use_cuda:
            yi = yi.cuda()

        mask = (yi < 0) # unobserved positions
        logits = logits[~mask]
        yi = yi[~mask]

        li = weight*F.binary_cross_entropy_with_logits(logits, yi) # loss for this sequence
        li.backward(retain_graph = True) # backprop to the embeddings

        loss += li.item()
        total += yi.size(0)

        # also calculate the recall and precision
        with torch.no_grad():
            p_hat = torch.sigmoid(logits)
            tp += torch.sum(p_hat*yi).item()
            gp += yi.sum().item()
            pp += p_hat.sum().item()


    # now, backprop the emebedding gradients through the model
    grad = z_detach.data.grad
    z.data.backward(grad)

    return loss, tp, gp, pp, total


def predict_cmap(model, x, y, use_cuda):
    b = len(x)
    #if use_cuda:
    #    x = [x_.cuda() for x_ in x]
    x,order = pack_sequences(x)
    if use_cuda:
        x = x.cuda()

    z = model.transform(x) # embed the sequences
    z = unpack_sequences(z, order)

    logits = []
    y_list = []
    for i in range(b):
        zi = z[i]
        lp = model.predict(zi.unsqueeze(0)).view(-1)

        yi = y[i].contiguous().view(-1)
        if use_cuda:
            yi = yi.cuda()
        mask = (yi < 0)

        lp = lp[~mask]
        yi = yi[~mask]

        logits.append(lp)
        y_list.append(yi)

    return logits, y_list


def eval_cmap(model, test_iterator, use_cuda):
    logits = []
    y = []

    for x,y_mb in test_iterator:
        logits_this, y_this = predict_cmap(model, x, y_mb, use_cuda)
        logits += logits_this
        y += y_this

    y = torch.cat(y, 0)
    logits = torch.cat(logits, 0)

    loss = F.binary_cross_entropy_with_logits(logits, y).item()

    p_hat = torch.sigmoid(logits)
    tp = torch.sum(y*p_hat).item()
    pr = tp/torch.sum(p_hat).item()
    re = tp/torch.sum(y).item()
    f1 = 2*pr*re/(pr + re)            

    y = y.cpu().numpy()
    logits = logits.data.cpu().numpy()

    aupr = average_precision(y, logits)

    return loss, pr, re, f1, aupr

def infinite_loop(it):
    while True:
        for x in it:
            yield x


def main():
    import argparse
    parser = argparse.ArgumentParser('Script for training multitask embedding model')

    # model hyperparameters/architecture settings

    # embedding model architecture
    parser.add_argument('--model', default='/hpc/group/naderilab/eleanor/prose_data/saved_models/_iter1000000.sav', help='pretrained model (optional)')
    parser.add_argument('--resume', action='store_true', help='resume training')

    parser.add_argument('--embedding-dim', type=int, default=100, help='embedding dimension (default: 100)')
    parser.add_argument('--dropout', type=float, default=0, help='dropout probability (default: 0)')


    # for the structural similarity prediction module
    parser.add_argument('--allow-insert', action='store_true', help='model insertions (default: false)')

    # training parameters

    parser.add_argument('-n', '--num-steps', type=int, default=1000, help='number ot training steps (default: 1,000)')
    parser.add_argument('--save-interval', type=int, default=100, help='frequency of saving (default:; 100)')

    parser.add_argument('--contacts', type=float, default=1, help='weight on the contact prediction task (default: 1)')
    parser.add_argument('--contacts-batch-size', type=int, default=100, help='minibatch size for contact maps (default: 50)')

    parser.add_argument('--weight-decay', type=float, default=0, help='L2 regularization (default: 0)')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate (default: 1e-4)')
    parser.add_argument('--clip', type=float, default=np.inf, help='gradient clipping max norm (default: inf)')
    parser.add_argument('--augment', type=float, default=0, help='resample amino acids during training with this probability (default: 0)')

    parser.add_argument('-o', '--output', help='output file path (default: stdout)')
    parser.add_argument('--save-prefix', help='path prefix for saving models')
    parser.add_argument('-d', '--device', type=int, default=-2, help='compute device to use')
    parser.add_argument('--seed', default=1357, type=int, help='random seed for replication')

    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    prefix = args.output
    d = args.device
    use_cuda = (d != -1) and torch.cuda.is_available()
    if d >= 0:
        torch.cuda.set_device(d)

    mi = 20
    ma = 1000
    root = '/hpc/group/naderilab/eleanor/Efficient_PLM/data/SCOPe/pdbstyle-2.06'
    path = '/hpc/group/naderilab/eleanor/Efficient_PLM/data/SCOPe/astral-scopedom-seqres-gd-sel-gs-bib-95-2.06.train.fa'
    contacts_train = ContactMapDataset(path, root=root, min_length=mi, max_length=ma)

    path = '/hpc/group/naderilab/eleanor/Efficient_PLM/data/SCOPe/astral-scopedom-seqres-gd-sel-gs-bib-95-2.06.test.fa'
    contacts_test = ContactMapDataset(path, root=root) #, max_length=max_length)
    num_steps = args.num_steps

    # data augmentation by resampling amino acids
    augment = None
    if args.augment > 0:
        augment = args.augment
        trans = torch.ones(21, 21)
        trans = trans/trans.sum(1, keepdim=True)
        if use_cuda:
            trans = trans.cuda()
        augment = MultinomialResample(trans, augment)
    print('# resampling amino acids with p:', args.augment, file=sys.stderr)

    contacts_train.augment = augment

    batch_size = args.contacts_batch_size

    cmap_train_iterator = torch.utils.data.DataLoader(contacts_train
                                                        , batch_size=batch_size
                                                        , shuffle=True
                                                        , collate_fn=collate_lists
                                                        )
    #batch_size = 4 # use smaller batch size for calculating test set results
    cmap_test_iterator = torch.utils.data.DataLoader(contacts_test
                                                    , batch_size=batch_size
                                                    , collate_fn=collate_lists
                                                    )

    ## initialize the model
    print('# using pretrained model:', args.model, file=sys.stderr)
    encoder = torch.load(args.model)
       
    resume = args.resume
    step = 0

    # encoder is multilayer LSTM with projection layer
    # replace projection layer for structure-based embeddings
    proj = encoder.proj
    encoder.cloze = proj  # keep the projection layer for the cloze task
    
    # make new projection layer for the structure embeddings
    embedding_size = args.embedding_dim

    n_hidden = proj.in_features
    proj = nn.Linear(n_hidden, embedding_size)
    encoder.proj = proj
    encoder.nout = embedding_size

    # contact map prediction parameters
    cmap_predict = BilinearContactMap(n_hidden)
    model = ProSEMT(encoder, None, cmap_predict)

    model.train()

    if use_cuda:
        model.cuda()

    ## setup training parameters and optimizer
    weight_decay = args.weight_decay
    lr = args.lr
    clip = args.clip

    print('# training with Adam: lr={}, weight_decay={}'.format(lr, weight_decay), file=sys.stderr)
    params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    #optim = torch.optim.Adagrad(params, lr=0.0003, weight_decay=weight_decay)

    ## train the model
    print('# training model', file=sys.stderr)

    save_prefix = args.save_prefix
    output = args.output
    if output is None:
        output = sys.stdout
    else:
        output = open(output, 'w')
    digits = int(np.floor(np.log10(num_steps))) + 1
    tokens = ['iter','rrc_loss', 'rrc_pr', 'rrc_re', 'rrc_f1', 'rrc_aupr']

    rrc = infinite_loop(cmap_train_iterator)
    #rrc = iter(cmap_train_iterator)
    cmap_n = 0
    cmap_loss_accum = 0
    cmap_pp = 0
    cmap_pr_accum = 0
    cmap_gp = 0
    cmap_re_accum = 0

    save_iter = 100
    save_interval = args.save_interval
    while save_iter <= step:
        save_iter = min(save_iter*10, save_iter+save_interval, num_steps) # next save

    for i in range(step, num_steps):
        c_x, c_y = next(rrc)
        loss, tp, gp_, pp_, b = cmap_grad(model, c_x, c_y, use_cuda, weight=1)

        cmap_gp += gp_
        delta = tp - gp_*cmap_re_accum
        cmap_re_accum += delta/cmap_gp

        cmap_pp += pp_
        delta = tp - pp_*cmap_pr_accum
        cmap_pr_accum += delta/cmap_pp

        cmap_n += b
        delta = b*(loss - cmap_loss_accum)
        cmap_loss_accum += delta/cmap_n


        # clip the gradients if needed
        if not np.isinf(clip):
            # only clip the RNN layers
            nn.utils.clip_grad_norm_(model.embedding.layers.parameters(), clip)

        # parameter update
        optim.step()
        optim.zero_grad()
        model.clip() # projected gradient for bounding ordinal regression parameters

        # report progressive results
        if (i+1) % 10 == 0:
            line = '# [{}/{}] training {:.1%} loss={:.5f}, precision={:.5f}, recall={:.5f}'
            line = line.format(i+1, num_steps, i/num_steps
                              , cmap_loss_accum, cmap_pr_accum, cmap_re_accum
                              )
            print(line, end='\r', file=sys.stderr)


        # evaluate and save model
        if i+1 == save_iter:
            save_iter = min(save_iter*10, save_iter+save_interval, num_steps) # next save
            tokens = ['loss', 'precision', 'recall', 'f1']
            line = '\t'.join(tokens)
            print(line, file=output)

            print(' '*80, end='\r', file=sys.stderr)
            f1 = 2*cmap_pr_accum*cmap_re_accum/(cmap_pr_accum + cmap_re_accum)
            tokens = [cmap_loss_accum, cmap_pr_accum, cmap_re_accum, f1, '-']
            tokens = [x if type(x) is str else '{:.5f}'.format(x) for x in tokens]
            line = '\t'.join([str(i+1).zfill(digits), 'train'] + tokens)
            print(line, file=output)
            output.flush()

            cmap_n = 0
            cmap_loss_accum = 0
            cmap_pp = 0
            cmap_pr_accum = 0
            cmap_gp = 0
            cmap_re_accum = 0

            # eval and save model
            model.eval()

            with torch.no_grad():
                cmap_loss, cmap_pr, cmap_re, cmap_f1, cmap_aupr = \
                        eval_cmap(model, cmap_test_iterator, use_cuda)

            tokens = ['loss', 'precision', 'recall', 'f1','aupr']
            line = '\t'.join(tokens)
            print(line, file=output)

            tokens = [cmap_loss, cmap_pr, cmap_re, cmap_f1, cmap_aupr]
            tokens = [x if type(x) is str else '{:.5f}'.format(x) for x in tokens]
            line = '\t'.join([str(i+1).zfill(digits), 'test'] + tokens)
            print(line, file=output)
            output.flush()


            # save the model
            if save_prefix is not None:
                save_path = save_prefix + '_iter' + str(i+1).zfill(digits) + '.sav'
                model.cpu()
                torch.save(model, save_path)
                if use_cuda:
                    model.cuda()

            # flip back to train mode
            model.train()


if __name__ == '__main__':
    main()