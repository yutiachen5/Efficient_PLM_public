import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from prose.ALLY.strategy import Strategy
from torch import nn
import sys
import wandb
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from copy import deepcopy
from torch.utils.data.dataset import TensorDataset
import pdb
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import pairwise_distances
from scipy import stats
from prose.ALLY.lambdautils import lambdanet, lambdaset
from torch.nn.utils.rnn import PackedSequence
from torch.autograd import detect_anomaly
from torch.utils.data import Subset
from prose.utils import pack_sequences, unpack_sequences
from prose.utils import LargeWeightedRandomSampler
from prose.utils import collate_seq2seq
from prose.datasets import FastaDataset, ClozeDataset, UnmaskedDataset
import time
import random
from datetime import datetime
from tqdm import tqdm 
import os 


class ALLYSampling(Strategy):
    def __init__(self, clf, use_cuda, base, opts):
        super(ALLYSampling, self).__init__(clf, use_cuda, base, opts)

        self.epsilon = self.opts['epsilon']
        self.lr_dual = self.opts['lr_dual']
        self.lambda_test_size = self.opts['lambdaTestSize']
        self.alg = "ALLY"
        self.use_cuda = use_cuda

    def query(self, rd, n):
        # Prepare data for lambdanet training
        X_train, X_test, y_train, y_test = self.prepare_data_lambda()
        print('Done with generating embedding of base set:', time.strftime("%H:%M:%S", time.localtime()))

        # Train Lambdanet
        self.train_test_lambdanet(X_train, X_test, y_train, y_test)
        print('Done with training lambdanet:', time.strftime("%H:%M:%S", time.localtime()))

        # Drop the seqs which have been seen by model but has low lambda from base set 
        if self.opts['base'] == 'drop':
            valid_mask = (self.idxs_base == 1) & (self.flag >= 1)
            sorted_lambda_indices = np.argsort(self.lambdas)
            valid_sorted_lambda_indices = sorted_lambda_indices[valid_mask[sorted_lambda_indices]]
            low_lambda_indices = valid_sorted_lambda_indices[:self.opts['nQuery']]
            drop_indices = [i for i, f in enumerate(self.flag) if i in low_lambda_indices]
            assert len(set(drop_indices)) == self.opts['nQuery']
            # update base set index
            self.idxs_base[drop_indices] = False

        held_data = Subset(self.all, self.held_indices)
        held_emb_data = UnmaskedDataset(held_data, self.held_indices)
        X_indices, X_embedding = self.get_embedding(held_emb_data)
        emb_dataset = UnmaskedDataset(X_embedding, X_indices)
        indices, preds = self.predict_lambdas(emb_dataset)
        print('Done with generating embedding and getting predicted lambdas of held-out set:', time.strftime("%H:%M:%S", time.localtime()))

        if self.opts['query_mode'] == 'active':
            sorted_pairs = sorted(zip(preds, indices), reverse=True)  
            sorted_preds, sorted_indices = zip(*sorted_pairs)  
            if self.opts['base'] == 'drop':
                chosen_preds = list(sorted_preds)[:len(drop_indices)]
                chosen_indices = list(sorted_indices)[:len(drop_indices)]
            elif self.opts['base'] == 'keep':
                chosen_preds = list(sorted_preds)[:self.opts['nQuery']]
                chosen_indices = list(sorted_indices)[:self.opts['nQuery']]
            else: raise ValueError

        elif self.opts['query_mode'] == 'random':
            pairs = list(zip(preds, indices))
            chosen_preds, chosen_indices = map(list, zip(*random.sample(pairs, n)))

        elif self.opts['query_mode'] == 'passive':
            sorted_pairs = sorted(zip(preds, indices), reverse=False)
            sorted_preds, sorted_indices = zip(*sorted_pairs)  
            if self.opts['base'] == 'drop':
                chosen_preds = list(sorted_preds)[:len(drop_indices)]
                chosen_indices = list(sorted_indices)[:len(drop_indices)]
            elif self.opts['base'] == 'keep':
                chosen_preds = list(sorted_preds)[:self.opts['nQuery']]
                chosen_indices = list(sorted_indices)[:self.opts['nQuery']]
            else: raise ValueError
        else: raise ValueError 
        
        print("Done selecting new batch.")

        return chosen_indices, chosen_preds

    def prepare_data_lambda(self):
        trained_indices = [i for i, f in enumerate(self.flag[self.idxs_base]) if f >= 1]
        filtered_data = Subset(self.all, trained_indices)  # train lambdanet only with sequences that have been seen by the model
        filtered_emb_data = UnmaskedDataset(filtered_data, trained_indices)  # no maksing for generating embedding
        X_indices, X_embedding = self.get_embedding(filtered_emb_data)
        y_lambdas = self.lambdas[trained_indices]

        if self.lambda_test_size > 0: # default 0
            X_train, X_test, y_train, y_test = train_test_split(X_embedding, y_lambdas, test_size=self.lambda_test_size, random_state = self.seed)
        else:
            X_train = X_embedding
            X_test = []
            y_train = y_lambdas
            y_test = []
        return X_train, X_test, y_train, y_test

    def _train_lambdanet(self, epoch, loader_tr, optimizer, scheduler):
        self.reg.train()
        mseFinal = 0.

        iterator = iter(loader_tr)

        for i in range(len(loader_tr)):
            x,y,i = next(iterator)
            x,y = Variable(x.cuda().float()), Variable(y.cuda().float()) 
            optimizer.zero_grad()
            out = self.reg(x)
            loss = F.mse_loss(out.squeeze(), y)
            loss.backward()
            mseFinal += loss.item()
            optimizer.step()
        scheduler.step()
        
        return mseFinal/len(loader_tr)

    def train_test_lambdanet(self, X_train, X_test, y_train, y_test):
        scheduler = optim.lr_scheduler.StepLR(self.optimizer_net, step_size = 1, gamma=0.95)
        loader_tr = DataLoader(lambdaset(X_train, X_test, y_train, y_test, train = True), 
                                batch_size = self.opts['lambdanet_batch_size'], shuffle = True, drop_last=True)

        mseThresh = 1e-3 #Add as argument

        self.reg.train()
        epoch = 1
        mseCurrent = 10.
        print_every = 10
        while (mseCurrent > mseThresh) and (epoch < 150): #default values for STL
            mseCurrent = self._train_lambdanet(epoch, loader_tr, self.optimizer_net, scheduler)
            if epoch%print_every==0:
                print(f"{epoch} Lambda training mse:  {mseCurrent:.3f}", flush=True)
            epoch += 1
        mseFinal = 0.

        # Test L if needed
        if self.lambda_test_size > 0:
            idx, P = self.predict_lambdas(X_test, y_test)
            mseTest = F.mse_loss(P, torch.tensor(y_test))           
            print(f"-----> Lambda test mse: {mseTest.item():.2f}\n", flush=True)
        return None
	
    def predict_lambdas(self, emb_dataset):
        loader_te = DataLoader(emb_dataset, batch_size = self.opts['lambdanet_batch_size'], shuffle = False, drop_last=True)

        self.reg.eval()
        idx = []
        lambda_pred = []       
        with torch.no_grad():
            for x, i in loader_te:
                x = Variable(x.cuda().float())
                out = self.reg(x)
                lambda_pred += out.squeeze().data.cpu()
                idx += i
        idx = [i.item() for i in idx]
        lambda_pred = [p.item() for p in lambda_pred]
        return idx, lambda_pred

    def cloze_grad(self, x, y):
        y = y.data

        if self.use_cuda:
            x = PackedSequence(x.data.cuda(), x.batch_sizes)
            y = y.cuda()

        mask = (y < 20) 
        loss = 0
        correct = 0
        n = mask.float().sum().item()
        t = len(x.data)
        unpacked_loss = torch.full_like(y, -1, dtype=torch.float)

        if n > 0:
            logits, emb = self.clf(x)  # emb: [length,3093]
            logits = logits.data[mask]  # only calculate loss for noised positions
            y = y[mask] 
            loss = F.cross_entropy(logits, y, reduction = 'none')
            unpacked_loss[mask] = loss

            _, y_hat = torch.max(logits, 1)
            correct = torch.sum((y == y_hat).float()).item() 
        return unpacked_loss, loss, correct, n, t

    def compute_avg_abs_gradient(self, model):
        total_grad = 0.0
        count = 0

        for param in model.parameters():
            if param.grad is not None:
                total_grad += param.grad.abs().sum().item()  # Sum of absolute values of gradients
                count += param.grad.numel()  # Count the total number of elements in the gradient tensor

        avg_abs_gradient = total_grad / count if count > 0 else 0.0
        return avg_abs_gradient

    def _PDCL(self, optimizer, scheduler, train_loader, train_indices): 
        self.clf.train()

        iterator = iter(train_loader)

        for i in tqdm(range(1,len(train_loader)+1)): 
            x, y, idxs, order = next(iterator)
            idxs = np.array(train_indices)[idxs]  # original index
            lambdas = self.lambdas[idxs]
            lambdas = torch.tensor(lambdas, requires_grad = False).cuda()
            self.flag[idxs] += 1

            optimizer.zero_grad() 

            unpacked_loss, loss, correct, b, t = self.cloze_grad(x, y)
            unpacked_loss = PackedSequence(unpacked_loss, x.batch_sizes)
            unpacked_loss = unpack_sequences(unpacked_loss, order)

            self.n += b 
            delta = b*(torch.mean(loss) - self.lossCurrent)
            self.lossCurrent += delta/self.n
            delta = correct - b*self.accFinal
            self.accFinal += delta/self.n
            self.token += t

            unpacked_loss_mean = torch.stack([x[x != -1].mean() for x in unpacked_loss]) 
            
            lagrangian = (unpacked_loss_mean*(1+lambdas)-lambdas*self.epsilon).nanmean() 
            lagrangian.backward()
            avg_grad = self.compute_avg_abs_gradient(self.clf)
            for p in filter(lambda p: p.grad is not None, self.clf.parameters()): p.grad.data.clamp_(min=-.15, max=.15)
            if not np.isinf(self.opts['clip']):
                nn.utils.clip_grad_norm_(self.clf.layers.parameters(), self.opts['clip'])

            optimizer.step()

            unpacked_loss_mean = torch.nan_to_num(unpacked_loss_mean, nan=self.epsilon)  # skip nan when updating dual variables  

            lambdas += self.lr_dual*(unpacked_loss_mean-self.epsilon)
            lambdas[lambdas < 0] = 0
            self.lambdas[idxs] = lambdas.detach().cpu()
            lambda_mean = np.mean(self.lambdas[idxs])  
            slack = torch.mean(unpacked_loss_mean-self.epsilon)

            if i%self.opts['validate_every'] == 1 or self.opts['validate_every'] == 1:
                perplexity = self.validate()
                self.clf.train()
                
            if i%self.opts['dual_lr_stepsize'] == 0:
                self.lr_dual = self.opts['dual_lr_gamma']*self.lr_dual

            scheduler.step()
            
            # wandb.log({'lambda_aver': lambda_mean, 'train loss': self.lossCurrent, 'train acc': self.accFinal, 'val perplexity': perplexity, 
            #             'avg abs grad': avg_grad, 'slack': slack, '# tokens': self.token}) # per batch
        self.lr_dual = self.opts['lr_dual']

        return self.lossCurrent, self.accFinal

    def build_train_loader(self, train_indices): 
        base_training_set = Subset(self.all, train_indices)
        L = np.array([len(x) for x in base_training_set])
        weight = np.maximum(L/self.opts['max_length'], 1)
        sampler = LargeWeightedRandomSampler(weight, self.opts['num_steps']*self.opts['batch_size'])

        counts = np.zeros(21)
        for x in base_training_set:
            v,c = np.unique(x.numpy(), return_counts=True) # v: unique element, c: number of times each unique item appears
            counts[v] = counts[v] + c 
        noise = counts/counts.sum() 
        noise = torch.from_numpy(noise)

        cloze_data = ClozeDataset(base_training_set, self.opts['p'], noise)
        loader = DataLoader(cloze_data, batch_size=self.opts['batch_size'], 
                                            sampler=sampler,
                                            collate_fn=collate_seq2seq)

        return loader

    def train(self):
        output_dir = self.opts['output'] + '/' + self.opts['name'] + '/' 
        os.makedirs(output_dir, exist_ok=True) 

        train_indices = np.arange(self.n_all)[self.idxs_base]
        train_loader = self.build_train_loader(train_indices)
        
        lossCurrent, accCurrent= self._PDCL(self.optimizer_clf, self.scheduler_clf, train_loader, train_indices)   

        # save model after every epoch
        output_path = output_dir + datetime.now().strftime('%Y%m%d_%H%M') +'.sav' 
        torch.save(self.best_model, output_path)

        perplexity = self.validate(self.val_loader)

        if perplexity < self.perplexity_best:
            self.perplexity_best = perplexity
            self.epochs_no_improve = 0
            self.best_model = deepcopy(self.clf)
            output_path = output_dir + datetime.now().strftime('%Y%m%d_%H%M') +'.sav' 
            torch.save([self.best_model, self.reg], output_path) 
        else:
            self.epochs_no_improve += 1

        print(f"training accuracy: {accCurrent:.2f} \tTraining loss: {lossCurrent:.2f} ", flush=True)
        print('Done with training and validation:', time.strftime("%H:%M:%S", time.localtime()))


        

