import numpy as np
import pandas as pd
from torch import nn
import random
import gc
import sys
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from copy import deepcopy
import pdb
from torch.utils.data.dataset import TensorDataset, Subset, ConcatDataset
from prose.datasets import FastaDataset, ClozeDataset, UnmaskedDataset
from prose.utils import collate_seq2seq, collate_seq2seq_unmasked
from prose.utils import LargeWeightedRandomSampler
from torch.nn.utils.rnn import PackedSequence
from prose.ALLY.lambdautils import lambdanet, lambdaset
from prose.utils import pack_sequences, unpack_sequences
import time
from tqdm import tqdm 


class Strategy:
    def __init__(self, model, use_cuda, base, opts):
        self.model = model
        self.use_cuda = use_cuda
        self.base = base
        self.opts = opts

        np.random.seed(self.opts['seed'])
        idxs_train = np.random.choice(np.arange(len(self.base)), size=self.opts['nStart'], replace=False)

        self.val_loader = self.build_val_loader()
        self.held_cat, self.held_indices_all = self.get_held_out_sets(self.opts['path_held'], self.opts['max_length']) 
        self.all = ConcatDataset([self.base, self.held_cat])
        self.held_indices = list(range(self.held_indices_all[0][0], self.held_indices_all[0][1]))  # current held-out set indices

        self.n_all = len(self.base) + len(self.held_cat)
        self.lambdas = np.zeros(self.n_all)
        self.flag = np.zeros(self.n_all)  # the number of times each seq has been selected for training
        self.idxs_base = np.zeros(self.n_all, dtype=bool)
        self.idxs_base[idxs_train] = True

        self.clf = self.model.apply(self.weight_reset).cuda() 
        self.reg = lambdanet(input_dim = self.clf.get_embedding_dim()).cuda() 
        self.optimizer_clf = optim.Adam(self.clf.parameters(), lr = opts['plr'], weight_decay=opts['weight_decay'])
        self.optimizer_net = optim.Adam(self.reg.parameters(), lr = opts['alr'], weight_decay=1e-2)
        self.scheduler_clf = optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer_clf, T_0=500, eta_min=opts['plr']/2) 

        self.accFinal = 0.
        self.lossCurrent = 0.
        self.n = 0
        self.epochs_no_improve = 0
        self.perplexity_best = 10000.
        self.best_model = None
        self.token = 0

    def get_held_out_sets(self, path, max_length):
        held_out_sets = [
            FastaDataset(path+'/diff_ur25_ur20.fasta', max_length=max_length),
            # FastaDataset(path+'/diff_ur30_ur25.fasta', max_length=max_length),
            # FastaDataset(path+'/diff_ur35_ur30.fasta', max_length=max_length),
            # FastaDataset(path+'/diff_ur40_ur35.fasta', max_length=max_length),
            # FastaDataset(path+'/diff_ur45_ur40.fasta', max_length=max_length),
            # FastaDataset(path+'/diff_ur50_ur45.fasta', max_length=max_length)
        ]
        held_out_cat = ConcatDataset(held_out_sets)
        held_out_indices = []
        start = len(self.base)
        for i in range(len(held_out_sets)):
            end = start + len(held_out_sets[i])
            held_out_indices.append((start, end))
            start = end

        return held_out_cat, held_out_indices

    def split_held_set(self, idxs_base):
        idxs_held = np.arange(self.n_pool)[~self.idxs_base] 
        split_indices = np.array_split(idxs_held, self.opts['nsplit'])
        return split_indices

    def build_val_loader(self):
        np.random.seed(self.opts['seed'])
        # val_fasta = FastaDataset('/hpc/group/naderilab/eleanor/prose_data/data/uniref50_0.1.fasta')
        val_fasta = FastaDataset('/hpc/group/naderilab/eleanor/Efficient_PLM/data/demo_val.fa')
        idxs_val = np.random.choice(np.arange(len(val_fasta)), size=self.opts['val_size'], replace=False) 
        val_fasta_subset = Subset(val_fasta, idxs_val)
        del val_fasta
        gc.collect()
        val_fasta_unmasked = UnmaskedDataset(val_fasta_subset, idxs_val) 
        loader = DataLoader(val_fasta_unmasked, batch_size=self.opts['batch_size'], collate_fn=collate_seq2seq_unmasked)
        return loader

    def weight_reset(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            m.reset_parameters()

    def query(self, n):
        pass

    def update(self, chosen_idx, chosen_preds, rd):
        # update base-set index
        self.idxs_base[chosen_idx] = True
        assert len(set(chosen_idx)) == self.opts['nQuery']

        # initialize the value of lambda of added samples as predicted lambda
        self.lambdas[chosen_idx] = chosen_preds

        # reinitialize flag 
        if self.opts['base'] == 'drop':
            self.flag = np.zeros(self.n_all)

        # update held-out set
        if self.opts['held_out'] == 'append': 
            remaining_idx = [i for i in self.held_indices if i not in chosen_idx]
            held_indices_next = list(range(self.held_indices_all[rd+1][0], self.held_indices_all[rd+1][1]))
            self.held_indices = held_indices_next + remaining_idx
        elif self.opts['held_out'] == 'discard': 
            self.held_indices = list(range(self.held_indices_all[rd+1][0], self.held_indices_all[rd+1][1]))
        else: raise Exception('wrong option for held-out sets')
            
    def validate(self, val_loader): 
        self.clf.eval()     
        iterator = iter(val_loader)

        perplexity = []
        with torch.no_grad():
            for i in range(len(val_loader)):
                x, idxs, order = next(iterator)
                logits = self.cloze_grad_val(x)
                probs = torch.softmax(logits.data, dim=1)
                probs_filtered = probs.cpu().gather(1, x.data.unsqueeze(1)).squeeze(1)
                probs_filtered = PackedSequence(probs_filtered, x.batch_sizes)
                probs_filtered = unpack_sequences(probs_filtered, order)
                seq_perplexity = [torch.exp(-torch.log(prob).mean()) for prob in probs_filtered]
                perplexity.extend(seq_perplexity)
                
        return torch.mean(torch.tensor(perplexity)) 

    def cloze_grad_val(self, x):
        if self.use_cuda:
            x = PackedSequence(x.data.cuda(), x.batch_sizes)
        logits, _ = self.clf(x)

        return logits

    def get_embedding(self, dataset):
        self.clf.eval()
        loader = DataLoader(dataset, batch_size=self.opts['emb_batch_size'], collate_fn=collate_seq2seq_unmasked) 
        iterator = iter(loader)
        idx = []
        embedding = []

        with torch.no_grad():
            for i in tqdm(range(len(loader))): 
                x, idxs, order = next(iterator)
                if self.use_cuda: x = PackedSequence(x.data.cuda(), x.batch_sizes)
                _, emb = self.clf(x)
                packed_emb = PackedSequence(emb, x.batch_sizes)  
                unpacked_emb = unpack_sequences(packed_emb, order) # unpack emb
                unpacked_emb = torch.stack([x.mean(dim=0) for x in unpacked_emb]) # [512,3093]
                idx += idxs
                embedding += unpacked_emb.detach().cpu().data 
                
        return idx, embedding
    