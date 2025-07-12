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
import torch.utils.data
from prose.utils import LargeWeightedRandomSampler
from prose.datasets import FastaDataset, ClozeDataset
from prose.ALLY.ally_tf import ALLYSampling
from torch.utils.data import Dataset, DataLoader, Subset
from prose.models.transformer import TransformerMLM
import time
import argparse
import random
import os

# taskID=int(os.environ['SLURM_ARRAY_TASK_ID'])
# jobName = str(os.environ['SLURM_JOB_NAME'])

def seed_everything(seed: int):    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def infinite_loop(it):
    while True:
        for x in it:
            yield x

def main():
    parser = argparse.ArgumentParser('Script for training the embedding model')

    # training dataset
    parser.add_argument('--path-train', default='/hpc/group/naderilab/navid/mmseqs/results/uniref20_rep_seq.fasta', help='path to training dataset in fasta format (default: data/uniprot/uniref90.fasta)')

    # embedding model architecture - Transformer
    parser.add_argument('--d-model', type=int, default=1024, help='the number of expected features in the encoder/decoder inputs (default=128)')
    parser.add_argument('--dim-feedforward', type=int, default=512, help='the dimension of the feedforward network model (default=512)')
    parser.add_argument('--nhead', type=int, default=4, help='the number of heads in the multiheadattention models (default=4)')
    parser.add_argument('--nlayer', type=int, default=3, help='the number of sub-encoder-layers in the encoder (default=3)')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout probability (default: 0.1)')

    # training parameters
    parser.add_argument('-n', '--num-steps', type=int, default=2000, help='number ot training steps (default: 2,000)')
    parser.add_argument('--max-length', type=int, default=1024, help='sample sequences down to this maximum length during training (default: 500)')
    parser.add_argument('-p', type=float, default=0.1, help='cloze residue masking rate (default: 0.1)')
    parser.add_argument('--batch-size', type=int, default=256, help='minibatch size (default: 256)')
    parser.add_argument('--weight-decay', type=float, default=0, help='L2 regularization (default: 0)')
    parser.add_argument('--plr', type=float, default=1e-4, help='learning rate (default: 1e-4) of PROSE')
    parser.add_argument('--clip', type=float, default=1e-4, help='gradient clipping max norm (default: inf)') 
    parser.add_argument('-o', '--output', help='output file path', default='/hpc/group/naderilab/eleanor/Efficient_PLM/saved_models/')
    parser.add_argument('-d', '--device', type=int, default=-2, help='compute device to use')
    parser.add_argument('--val-size', type=int, default=10000, help='number of validation samples')
    parser.add_argument('--debug', action='store_true')

    # ally
    parser.add_argument('--seed', help='random seed', type=int, default=1357)
    parser.add_argument('--query-mode', type=str, default='active', help='query mode of ally sampling: largest lambda (active), random lambda (random), and least lambda (passive)')
    parser.add_argument('--nQuery', help='number of points to drop and add in each round', type=int, default=10000)
    parser.add_argument('--nStart', help='number of points to start', type=int, default=50000)
    parser.add_argument('--alr', help='learning rate of lambdanet', type=float, default=1e-3)
    parser.add_argument('--validate-every', help = 'validate every n steps', type = int, default = 10)
    parser.add_argument('--epsilon', help='constant tightness for constrained learning', type=float, default=2.75) 
    parser.add_argument('--lr-dual', help='dual learning rate', type=float, default=0.1)
    parser.add_argument('--lambdaValSize', help = 'Size in percentage of validation set for lambda net', type = float, default = 0.2)
    parser.add_argument('--lambdanet-batch-size', help='lambda net batch size', type=int, default=64)
    parser.add_argument('--emb-batch-size', help='get embedding batch size', type=int, default=512)
    parser.add_argument('--path-held',help='path of held-out set',default='/hpc/group/naderilab/navid/mmseqs')
    parser.add_argument('--held-out', help='option to append or discard remaining samples in held-out after each round', type=str, default='discard')
    parser.add_argument('--base', help='option to keep or drop low-lambda samples after each round', type=str, default='keep')
    parser.add_argument('--dual_lr_gamma', type=float, default=0.5, help='rate of dual lr decay')
    parser.add_argument('--dual_lr_stepsize', type=int, default=500, help='stepsize of dual lr schedule')
    parser.add_argument('--cluster', help='How to cluster for diversity in primaldual', type = str, default='nocluster')
    parser.add_argument('--nClusters', help='the number of clusters', type = int, default=4000)
    parser.add_argument('--pooling', help='pooling method', type = str, default='mean')
    parser.add_argument('--alpha-slack', type=float, help="slack alpha", default=0.1)
    parser.add_argument('--lr-slack', help='slack learning rate', type=float, default=0.01)


    parser.add_argument('--name', type=str, default='test_run', help='name of the run for saving to wandb')

    args = parser.parse_args()

    # parameter search

    # all_configs = [(plr, s) for plr in [1e-5,5e-5] for s in [100, 200, 300]]
    # args.plr, args.seed = all_configs[taskID]
    # args.name = 'plr_'+str(args.plr)+'_nly_'+str(args.nlayer)+'_d_'+str(args.d_model)+'_s_'+str(args.seed)+'_ffd_'+str(args.dim_feedforward)+'_e_'+str(args.epsilon)

    # all_configs = [(l, ffd, s) for l in [5, 6] for ffd in [128, 256] for s in [100, 200, 300]]
    # args.nlayer, args.seed, args.dim_feedforward = all_configs[taskID] 
    # args.name = 'plr_1e-3_nly_'+str(args.nlayer)+'_d_'+str(args.d_model)+'_s_'+str(args.seed)+'_ffd_'+str(args.dim_feedforward)

    # all_configs = [(l, nc, s) for l in [5, 6] for nc in [2000,4000,6000,8000,10000] for s in [100, 200, 300]]
    # args.nlayer, args.nClusters, args.seed = all_configs[taskID] 
    # args.name = 'plr_'+str(args.plr)+'_nly_'+str(args.nlayer)+'_d_'+str(args.d_model)+'_s_'+str(args.seed)+'_ffd_'+str(args.dim_feedforward)

    # all_configs = [(slr, a) for slr in [0.1, 0.5, 0.8] for a in [0.5, 1, 2]]
    # args.slr, args.alpha = all_configs[taskID]
    # args.name = 'a_'+str(args.slr)+'_slr_'+str(args.alpha)+'_esm2'

    seed_everything(args.seed)

    # track pre-training progress by weights & bias
    wandb.init(
        project="array_jobs2",
        name=args.name, 
        config=vars(args) 
    )

    # set the device
    d = args.device
    use_cuda = (d != -1) and torch.cuda.is_available()
    if d >= 0:
        torch.cuda.set_device(d)

    # load the dataset
    base = FastaDataset(args.path_train, max_length=args.max_length, debug=args.debug)
    print('Done with loading base set data:', time.strftime("%H:%M:%S", time.localtime()))

    # set output path
    opts = vars(args)
    if args.output is None:
        opts['output'] = sys.stdout
    else:
        opts['output'] = args.output


    # weight each sequence by the number of fragments
    L = np.array([len(x) for x in base.x])
    weight = np.maximum(L/args.max_length, 1)
    sampler = LargeWeightedRandomSampler(weight, args.batch_size * args.num_steps)

    # initialize embedding model
    model = TransformerMLM(input_dim=21, emb_dim=args.d_model, num_heads=args.nhead, num_layers=args.nlayer, 
                            dim_feedforward=args.dim_feedforward, dropout=args.dropout, out_dim=21, max_len=args.max_length)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('# params: ', total_params)
    print('# trainable params: ', trainable_params)

    step = 0
    model.train()
    if use_cuda:
        model.cuda()

    print('# training with Adam: primal lr={}, dual lr={}'.format(args.alr, args.lr_dual), file=sys.stderr)
    print('# training model', file=sys.stderr)
    print('number of samples in base set: {}'.format(args.nStart), flush=True)

    # initialize ALLY for constrained learning
    alg = ALLYSampling(model, use_cuda, base, opts)

    NUM_ROUND = 7  # 6 held-out sets + base set
    
    sampled = []
    for rd in range(0, NUM_ROUND):
        print('Round {}'.format(rd), flush=True)
        alg.train()
        torch.cuda.empty_cache()
        gc.collect()

        if rd < NUM_ROUND - 1:
            # query current held-out set
            chosen_indices, chosen_preds = alg.query(rd, args.nQuery)
            sampled += list(chosen_indices)

        if rd < NUM_ROUND - 2:
            # update current base set
            alg.update(chosen_indices, chosen_preds, rd)

        nsamples = args.nStart + (rd+1)*args.nQuery

        if sum(~alg.idxs_base) < args.nQuery: 
            sys.exit('Too few remaining points to query')

if __name__ == '__main__':
    main()
