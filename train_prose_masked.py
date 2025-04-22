from __future__ import print_function,division

import numpy as np
import wandb
import sys
import os
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# from torch.nn.utils.rnn import PackedSequence
import torch.utils.data
from prose.utils import collate_seq2seq
from prose.utils import LargeWeightedRandomSampler
from prose.datasets import FastaDataset, ClozeDataset
from prose.ALLY.ally import ALLYSampling
from prose.ALLY.dataset import get_handler
from torch.utils.data import Dataset, DataLoader, Subset
from prose.models.lstm import SkipLSTM
import time
import argparse


def infinite_loop(it):
    while True:
        for x in it:
            yield x

def main():
    parser = argparse.ArgumentParser('Script for training multitask embedding model')

    # training dataset
    parser.add_argument('--path-train', default='/hpc/group/naderilab/navid/mmseqs/results/uniref20_rep_seq.fasta', help='path to training dataset in fasta format (default: data/uniprot/uniref90.fasta)')

    # embedding model architecture - LSTM
    parser.add_argument('--resume', action='store_true', help='resume training')
    parser.add_argument('--rnn-dim', type=int, default=512, help='hidden units of RNNs (default: 512)')
    parser.add_argument('--num-layers', type=int, default=3, help='number of RNN layers (default: 3)')
    parser.add_argument('--dropout', type=float, default=0, help='dropout probability (default: 0)')

    # training parameters
    parser.add_argument('-n', '--num-steps', type=int, default=2000, help='number ot training steps (default: 2,000)')
    parser.add_argument('--max-length', type=int, default=500, help='sample sequences down to this maximum length during training (default: 500)')
    parser.add_argument('-p', type=float, default=0.1, help='cloze residue masking rate (default: 0.1)')
    parser.add_argument('--batch-size', type=int, default=256, help='minibatch size (default: 256)')
    parser.add_argument('--weight-decay', type=float, default=0, help='L2 regularization (default: 0)')
    parser.add_argument('--plr', type=float, default=0.0001, help='learning rate (default: 1e-4) of PROSE')
    parser.add_argument('--clip', type=float, default=np.inf, help='gradient clipping max norm (default: inf)') 
    parser.add_argument('-o', '--output', help='output file path', default='/hpc/group/naderilab/eleanor/Efficient_PLM/saved_models')
    parser.add_argument('-d', '--device', type=int, default=-2, help='compute device to use')
    parser.add_argument('--val-size', type=int, default=100000, help='number of validation samples')
    parser.add_argument('--debug', action='store_true')

    # ally
    parser.add_argument('--seed', help='random seed', default=1357)
    parser.add_argument('--query-mode', type=str, default='active', help='query mode of ally sampling: largest lambda (active), random lambda (random), and least lambda (passive)')
    parser.add_argument('--nClasses', default=1)
    parser.add_argument('--nQuery', help='number of points to drop and add in each round', type=int, default=10000)
    parser.add_argument('--nStart', help='number of points to start', type=int, default=50000)
    parser.add_argument('--alr', help='ALLY primal learning rate', type=float, default=1e-3)
    parser.add_argument('--validate-every', help = 'validate every n steps', type = int, default = 10)
    parser.add_argument('--redund', help='redundancy in stl', type=int, default=0)
    parser.add_argument('--epsilon', help='constant tightness', type=float, default=2.75) 
    parser.add_argument('--cluster', help='How to cluster for diversity in primaldual', type = str, default='nocluster')
    parser.add_argument('--nPrimal', help='number of primal steps', type=int, default=1)
    parser.add_argument('--lr-dual', help='ALLY dual learning rate', type=float, default=0.1)
    parser.add_argument('--lambdaTestSize', help = 'Size in percentage of test set for lambda net', type = float, default = 0)
    parser.add_argument('--lambdanet-batch-size', help='lambda net batch size', type=int, default=64)
    parser.add_argument('--emb-batch-size', help='get embedding batch size', type=int, default=512)
    parser.add_argument('--nsplit', help='number of split of held-out set', type=int, default=6)
    parser.add_argument('--path-held',help='path of held-out set',default='/hpc/group/naderilab/navid/mmseqs')
    parser.add_argument('--held-out', help='option to append or discard remaining samples in held-out after each round', type=str, default='discard')
    parser.add_argument('--base', help='option to keep or drop low-lambda samples after each round', type=str, default='keep')
    parser.add_argument('--dual_lr_gamma', type=float, default=0.5, help='rate of dual lr decay')
    parser.add_argument('--dual_lr_stepsize', type=int, default=500, help='stepsize of dual lr schedule')

    parser.add_argument('--name', type=str, default='test_run', help='name of the run for saving to wandb')

    args = parser.parse_args()
    epsilon = args.epsilon
    dual_lr = args.lr_dual
    alr = args.alr


    # initiate wandb
    # wandb.init(
    #     project="Prose_MLM",
    #     name=args.name, 
    #     config=vars(args) 
    # )

    ## set the device
    d = args.device
    use_cuda = (d != -1) and torch.cuda.is_available()
    if d >= 0:
        torch.cuda.set_device(d)

    # load the dataset
    max_length = args.max_length # language modeling sequences to have this maximum length for limiting memory usage during training

    base = FastaDataset(args.path_train, max_length=max_length, debug=args.debug)
    print('Done with loading base set data:', time.strftime("%H:%M:%S", time.localtime()))

    opts = vars(args)
    if args.output is None:
        opts['output'] = sys.stdout
    else:
        opts['output'] = args.output

    # make the minbatch iterators
    num_steps = args.num_steps
    batch_size = args.batch_size

    # weight each sequence by the number of fragments
    L = np.array([len(x) for x in base.x])
    weight = np.maximum(L/max_length, 1)
    sampler = LargeWeightedRandomSampler(weight, batch_size*num_steps)

    model = SkipLSTM(nin=21, nout=21, hidden_dim=args.rnn_dim, num_layers=args.num_layers, dropout=args.dropout)

    step = 0
    model.train()
    if use_cuda:
        model.cuda()

    # setup training parameters 
    print('# training with Adam: primal lr={}, dual lr={}'.format(alr, dual_lr), file=sys.stderr)
    # train the model
    print('# training model', file=sys.stderr)
    # model.train()
    print('number of samples in base set: {}'.format(args.nStart), flush=True)

    alg = ALLYSampling(model, use_cuda, base, opts)

    NUM_ROUND = 7  # 6 held-out sets 
    
    sampled = []
    for rd in range(0, NUM_ROUND):
        print('Round {}'.format(rd), flush=True)
        alg.train()
        torch.cuda.empty_cache()
        gc.collect()

        if rd < NUM_ROUND - 1:
            # Query
            chosen_idx, chosen_preds = alg.query(rd, args.nQuery)
            sampled += list(chosen_idx)

        if rd < NUM_ROUND - 2:
            # Update
            alg.update(chosen_idx, chosen_preds, rd)

        nsamples = args.nStart + (rd+1)*args.nQuery

        if sum(~alg.idxs_base) < args.nQuery: 
            sys.exit('Too few remaining points to query')

if __name__ == '__main__':
    main()
