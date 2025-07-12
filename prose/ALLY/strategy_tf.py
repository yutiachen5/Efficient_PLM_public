import numpy as np
import pandas as pd
from torch import nn
import random
import gc
import sys
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import TensorDataset, Subset, ConcatDataset
from prose.datasets import FastaDataset, ValPPLDataset, UnmaskedDataset
from prose.utils import pad_seq_val, pad_seq_emb
from prose.utils import LargeWeightedRandomSampler
from torch.nn.utils.rnn import PackedSequence
from prose.ALLY.lambdautils import lambdanet, lambdaset
from prose.utils import pack_sequences, unpack_sequences
import time
from tqdm import tqdm 
from prose.ALLY.swe import Interp1d, SWE_Pooling


class Strategy:
    def __init__(self, model, use_cuda, base, opts):
        self.model = model
        self.use_cuda = use_cuda
        self.base = base
        self.opts = opts

        np.random.seed(self.opts['seed'])
        idxs_train = np.random.choice(np.arange(len(self.base)), size=self.opts['nStart'], replace=False)

        self.val_loader = self.build_val_loader(self.opts['max_length'])
        self.held_cat, self.held_indices_all = self.get_held_out_sets(self.opts['path_held'], self.opts['max_length']) 
        self.all = ConcatDataset([self.base, self.held_cat])
        self.held_indices = list(range(self.held_indices_all[0][0], self.held_indices_all[0][1]))  # current held-out set indices

        self.n_all = len(self.base) + len(self.held_cat)
        self.lambdas = torch.zeros(self.n_all, requires_grad=False)
        self.flag = np.zeros(self.n_all)  # the number of times each seq has been selected for training
        self.idxs_base = np.zeros(self.n_all, dtype=bool)
        self.idxs_base[idxs_train] = True
        self.slacks = torch.zeros(self.n_all, requires_grad=False) 

        self.clf = self.model.apply(self.weight_reset).cuda() 
        self.reg = lambdanet(input_dim = self.clf.get_embedding_dim()).cuda() 
        print(self.reg)
        
        self.optimizer_clf = optim.Adam(self.clf.parameters(), lr = opts['plr'], weight_decay=opts['weight_decay'])
        self.optimizer_net = optim.Adam(self.reg.parameters(), lr = opts['alr'])
        self.scheduler_clf = optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer_clf, T_0=500, eta_min=opts['plr']/2) 

        self.pooling = SWE_Pooling(d_in=self.opts['d_model'], num_slices=self.opts['d_model'], 
                            num_ref_points=self.opts['max_length'], freeze_swe=False)

        self.accFinal = 0.
        self.lossCurrent = 0.
        self.n = 0
        self.epochs_no_improve = 0
        self.perplexity_best = 10000.
        self.best_model = None
        self.token = 0

    def get_held_out_sets(self, path, max_length):
        held_out_sets = [
            # FastaDataset('/hpc/group/naderilab/eleanor/Efficient_PLM/data/demo_val.fa', max_length=max_length)
            FastaDataset(path+'/diff_ur25_ur20.fasta', max_length=max_length),
            FastaDataset(path+'/diff_ur30_ur25.fasta', max_length=max_length),
            FastaDataset(path+'/diff_ur35_ur30.fasta', max_length=max_length),
            FastaDataset(path+'/diff_ur40_ur35.fasta', max_length=max_length),
            FastaDataset(path+'/diff_ur45_ur40.fasta', max_length=max_length)
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

    def build_val_loader(self, max_length):
        np.random.seed(self.opts['seed'])
        val_fasta = FastaDataset('/hpc/group/naderilab/eleanor/prose_data/data/uniref50_0.1.fasta', max_length=max_length)
        # val_fasta = FastaDataset('/hpc/group/naderilab/eleanor/Efficient_PLM/data/demo_val.fa', max_length=max_length)

        idxs_val = np.random.choice(np.arange(len(val_fasta)), size=self.opts['val_size'], replace=False) 
        val_fasta_subset = Subset(val_fasta, idxs_val)
        del val_fasta
        gc.collect()

        L = np.array([len(x) for x in val_fasta_subset])
        weight = np.maximum(L/self.opts['max_length'], 1)
        sampler = LargeWeightedRandomSampler(weight, self.opts['num_steps']*self.opts['batch_size'])
        counts = np.zeros(21)
        for x in val_fasta_subset:
            v,c = np.unique(x.numpy(), return_counts=True) # v: unique element, c: number of times each unique item appears
            counts[v] = counts[v] + c 
        noise = counts/counts.sum() 
        noise = torch.from_numpy(noise)

        val_data = ValPPLDataset(val_fasta_subset, noise)
        loader = DataLoader(val_data, batch_size=self.opts['batch_size'], 
                                            sampler=sampler,
                                            collate_fn=pad_seq_val)
        # val_fasta_unmasked = UnmaskedDataset(val_fasta_subset, idxs_val) 
        # loader = DataLoader(val_fasta_unmasked, batch_size=self.opts['batch_size'], collate_fn=pad_seq_val)
        return loader

    def weight_reset(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            m.reset_parameters()

    def query(self, n):
        pass

    def update(self, chosen_indices, chosen_preds, rd):
        # update base-set index
        self.idxs_base[chosen_indices] = True
        assert len(set(chosen_indices)) == self.opts['nQuery']

        # initialize the value of lambda of added samples as predicted lambda
        self.lambdas[chosen_indices] = torch.tensor(chosen_preds)

        # reinitialize flag 
        # if self.opts['base'] == 'drop':
        #     self.flag = np.zeros(self.n_all)

        # update held-out set
        if self.opts['held_out'] == 'append': 
            remaining_idx = [i for i in self.held_indices if i not in chosen_indices]
            held_indices_next = list(range(self.held_indices_all[rd+1][0], self.held_indices_all[rd+1][1]))
            self.held_indices = held_indices_next + remaining_idx
        elif self.opts['held_out'] == 'discard': 
            self.held_indices = list(range(self.held_indices_all[rd+1][0], self.held_indices_all[rd+1][1]))
        self.reg = self.reg.apply(self.weight_reset).cuda() # reinitialize lambdanet after each round
            
    def validate(self): 
        self.clf.eval()     
        iterator = iter(self.val_loader)

        ppl = []
        with torch.no_grad():
            for i in range(len(self.val_loader)):
                x, padding_mask, _ = next(iterator) # padding_mask: [batch_size, max_len]
                logits, _ = self.clf(x.cuda(), padding_mask.cuda()) # logits: [batch_size, max_len, 21]

                mask = padding_mask.nonzero(as_tuple=True)  # (batch_idx, seq_idx)
                logits = logits[mask]  # [valid_pos, 21]
                probs = torch.softmax(logits, dim=1).cpu()  # [valid_pos, 21]
                x = x[mask]  # [valid_pos, 21]
                x = torch.argmax(x, dim=1) # [valid_pos]
                token_probs = probs.gather(1, x.long().unsqueeze(1)).squeeze(1)  # [valid_pos]
                batch_ppl = torch.exp(-torch.log(token_probs).mean())  # Scalar value
                ppl.append(batch_ppl)
                                
        return torch.stack(ppl).mean()

    def validate_esm(self): # follow the way ESM-2 calculates val ppl
        self.clf.eval()
        iterator = iter(self.val_loader)

        ppl = []
        with torch.no_grad():
            x_mod, x_orig, padding_indicator, masking_indicator = next(iterator) # both x_mod and x_orig are padded, x_orig: [batch_size, max_len]
            logits, _ = self.clf(x_mod.cuda(), padding_indicator.cuda())
            logits = logits.cpu()

            valid_pos = (masking_indicator != -1) & (padding_indicator == 1) # 1: unpadded, -1: unmodified 
            # print(valid_pos.shape)
            # print(x_orig)
            # print(type(x_orig))
            # print(x_orig.shape)
            # raise Exception
            x_valid = x_orig[valid_pos]
            logits_valid = logits[valid_pos]

            probs = torch.softmax(logits_valid, dim=1)
            token_probs = probs.gather(1, x_valid.long().unsqueeze(1)).squeeze(1) 
            batch_ppl = torch.exp(-torch.log(token_probs).mean())

            ppl.append(batch_ppl)

        return torch.stack(ppl).mean()

    def get_embedding(self, dataset):
        self.clf.eval()
        loader = DataLoader(dataset, batch_size=self.opts['emb_batch_size'], collate_fn=pad_seq_emb) 
        iterator = iter(loader)
        idx = []
        embedding = []

        with torch.no_grad():
            for i in tqdm(range(len(loader))): 
                x, padding_mask, idxs = next(iterator) # padding_mask: [batch_size, max_len]
                _, emb = self.clf(x.cuda(), padding_mask.cuda()) # emb: [batch_size, max_len, emb_dim]
                emb = emb.detach().cpu()

                if self.opts['pooling'] == 'swe':
                    pooled_emb = self.pooling(emb, padding_mask) # [batch_size, emb_dim]
                else: # mean pooling
                    pooled_emb = torch.sum(emb*padding_mask.unsqueeze(-1), dim=1)/torch.sum(padding_mask, dim=1, keepdim=True) # [batch_size, emb_dim]

                idx += idxs
                embedding += [pooled_emb[j] for j in range(pooled_emb.shape[0])]   

                
        return idx, embedding
    