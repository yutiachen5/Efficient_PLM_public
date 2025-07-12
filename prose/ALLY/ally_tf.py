import numpy as np
import pandas as pd
import time
import random
from datetime import datetime
from tqdm import tqdm 
from copy import deepcopy
import os 
import sys
import wandb
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from scipy import stats
from sklearn.cluster import MiniBatchKMeans
from sklearn.model_selection import train_test_split
from prose.ALLY.strategy_tf import Strategy
from prose.ALLY.lambdautils import lambdanet, lambdaset
from torch.utils.data import Subset
from prose.utils import LargeWeightedRandomSampler
from prose.utils import pad_seq
from prose.datasets import FastaDataset, ClozeDataset, UnmaskedDataset
from sklearn.preprocessing import MinMaxScaler


class ALLYSampling(Strategy):
    def __init__(self, clf, use_cuda, base, opts):
        super(ALLYSampling, self).__init__(clf, use_cuda, base, opts)

        self.epsilon = self.opts['epsilon']
        self.lr_dual = self.opts['lr_dual']
        self.lr_slack = self.opts['lr_slack']
        self.alpha = self.opts['alpha_slack']
        self.alg = "ALLY"
        self.use_cuda = use_cuda

    def query(self, rd, n):
        # Prepare data for lambdanet training
        X_train, X_test, y_train, y_test = self.prepare_data_lambda(rd)
        print('Done with generating embedding of base set:', time.strftime("%H:%M:%S", time.localtime()))

        scaler = MinMaxScaler() # scale lambdas
        y_train = scaler.fit_transform(y_train.reshape(-1, 1))
        y_test = scaler.transform(y_test.reshape(-1, 1)) 

        # Train Lambdanet
        self.train_val_lambdanet(X_train, X_test, y_train, y_test)
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

        # save embeddings for regression head architecture exploring 
        # np.save('/hpc/group/naderilab/eleanor/Efficient_PLM/prose/ALLY/saved_emb/'+self.opts['name']+'_rd_'+str(rd)+'_held.npy', torch.stack(X_embedding).numpy())

        emb_dataset = UnmaskedDataset(X_embedding, X_indices)
        indices, preds = self.predict_lambdas(emb_dataset, scaler)
        print('Done with generating embedding and getting predicted lambdas of held-out set:', time.strftime("%H:%M:%S", time.localtime()))


        if self.opts['query_mode'] == 'active':
            # Select samples with highest predicted lambda from each cluster to diversify samples
            if self.opts['cluster'] == "kmeans":
                # MiniBatch K-means on embeddings
                print("Clustering ....")
                kmeans = MiniBatchKMeans(n_clusters = self.opts['nClusters'], random_state = self.opts['seed'], 
                                        batch_size=self.opts['lambdanet_batch_size'], n_init='auto')
                cluster_indices = kmeans.fit_predict(X_embedding)
                sorted_triplets = sorted(zip(preds, indices, cluster_indices), reverse=True)  # descendingly 
                sorted_preds, sorted_indices, sorted_cluster_indices = zip(*sorted_triplets)
        
                chosen_indices = []
                chosen_preds = []
                indices_no_space_left = []
                preds_no_space_left = []
                space_in_clust = np.zeros(self.opts['nClusters'])+self.opts['nQuery']//self.opts['nClusters']

                for i, idx in enumerate(sorted_indices):
                    cluster_id = sorted_cluster_indices[i]
                    if space_in_clust[cluster_id] > 0:
                        chosen_indices.append(idx)
                        chosen_preds.append(sorted_preds[i])
                        space_in_clust[cluster_id] -= 1
                    else: 
                        indices_no_space_left.append(idx)
                        preds_no_space_left.append(sorted_preds[i])

                    if len(chosen_indices) == self.opts['nQuery']:
                        break  
                    if i == len(sorted_indices) - 1 and len(chosen_indices) < self.opts['nQuery']:
                        remaining_slots = self.opts['nQuery'] - len(chosen_indices)
                        chosen_indices += indices_no_space_left[:remaining_slots]
                        chosen_preds += preds_no_space_left[:remaining_slots]
                df = pd.DataFrame({'sorted_preds':sorted_preds, 'sorted_idxs':sorted_indices})
                # df.to_csv('/hpc/group/naderilab/eleanor/Efficient_PLM/prose/ALLY/saved_emb/'+self.opts['name']+'_rd_'+str(rd)+'_pred.csv', index = False)
            else:
                sorted_pairs = sorted(zip(preds, indices), reverse=True)
                sorted_preds, sorted_indices = zip(*sorted_pairs)  
                chosen_preds = list(sorted_preds)[:self.opts['nQuery']] # no diversity
                chosen_indices = list(sorted_indices)[:self.opts['nQuery']]

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
        
        print("Done selecting new batch.")

        return chosen_indices, chosen_preds

    def prepare_data_lambda(self, rd):
        trained_indices = [i for i, f in zip(np.arange(self.n_all)[self.idxs_base], self.flag[self.idxs_base]) if f >= 1]
        filtered_data = Subset(self.all, trained_indices)  # train lambdanet only with sequences that have been seen by the model

        filtered_emb_data = UnmaskedDataset(filtered_data, trained_indices)  
        X_indices, X_embedding = self.get_embedding(filtered_emb_data)
        # np.save('/hpc/group/naderilab/eleanor/Efficient_PLM/prose/ALLY/saved_emb/'+self.opts['name']+'_rd_'+str(rd)+'_base.npy', torch.stack(X_embedding).numpy())

        y_lambdas = self.lambdas[trained_indices]
        df = pd.DataFrame({'lambdas_trained':y_lambdas})
        # df.to_csv('/hpc/group/naderilab/eleanor/Efficient_PLM/prose/ALLY/saved_emb/'+self.opts['name']+'_rd_'+str(rd)+'_base.csv', index = False)

        if self.opts['lambdaValSize'] > 0: # default 0.2
            X_train, X_test, y_train, y_test = train_test_split(X_embedding, y_lambdas, test_size=self.opts['lambdaValSize'], random_state = self.opts['seed'])
        else:
            X_train = X_embedding
            X_test = []
            y_train = y_lambdas
            y_test = []
        return X_train, X_test, y_train, y_test

    def _train_lambdanet(self, epoch, loader_tr, optimizer):
        self.reg.train()
        mseFinal = 0.
        iterator = iter(loader_tr)

        for i in range(len(loader_tr)):
            x, y, i = next(iterator)
            x, y = Variable(x.cuda().float()), Variable(y.cuda().float()) 
            optimizer.zero_grad()
            out = self.reg(x) 
            loss = F.mse_loss(out.squeeze(), y.squeeze())
            loss.backward() 
            mseFinal += loss.item()
            optimizer.step()
        
        return mseFinal/len(loader_tr) # batch train mse

    def _val_lambdanet(self, loader_val):
        self.reg.eval()
        mseFinal = 0.
        iterator = iter(loader_val)

        for i in range(len(loader_val)):
            x, y, i = next(iterator)
            x, y = Variable(x.cuda().float()), Variable(y.cuda().float()) 
            out = self.reg(x) 
            loss = F.mse_loss(out.squeeze(), y.squeeze())
            mseFinal += loss.item()
        return mseFinal/len(loader_val) # batch val mse

    def train_val_lambdanet(self, X_train, X_test, y_train, y_test):
        best_reg = None
        scheduler = optim.lr_scheduler.StepLR(self.optimizer_net, step_size = 1, gamma=0.95)
        loader_tr = DataLoader(lambdaset(X_train, X_test, y_train, y_test, train = True), 
                                batch_size = self.opts['lambdanet_batch_size'], shuffle = True, drop_last=True)
        if self.opts['lambdaValSize'] > 0:
            loader_val = DataLoader(lambdaset(X_train, X_test, y_train, y_test, train = False), 
                                batch_size = self.opts['lambdanet_batch_size'], shuffle = True, drop_last=True)

        self.reg.train()
        mseBest = 1000.
        print_every = 1
        for epoch in range(100):
            mseCurrent = self._train_lambdanet(epoch, loader_tr, self.optimizer_net) 
            scheduler.step()

            if self.opts['lambdaValSize'] > 0:
                mseVal = self._val_lambdanet(loader_val)
                if mseVal < mseBest:
                    mseBest = mseVal
                    best_reg = deepcopy(self.reg)
            if epoch%print_every==0:
                if self.opts['lambdaValSize'] > 0:
                    print(f"{epoch} Lambda training mse:  {mseCurrent:.6f} | Lambda validation mse: {mseVal:.6f}", flush=True) # scaled
                else:
                    print(f"{epoch} Lambda training mse:  {mseCurrent:.6f}", flush=True) # scaled
            
        if self.opts['lambdaValSize'] > 0: # save best model in terms of minimum val mse
            self.reg = best_reg

        return None
	
    def predict_lambdas(self, emb_dataset, scaler):
        loader = DataLoader(emb_dataset, batch_size = self.opts['lambdanet_batch_size'], shuffle = False, drop_last=True)

        self.reg.eval()
        idx = []
        lambda_pred = []       
        with torch.no_grad():
            for x, i in loader:
                x = Variable(x.cuda().float())
                out = self.reg(x)
                lambda_pred += out.squeeze().data.cpu()
                idx += i
        idx = [i.item() for i in idx]
        lambda_pred = [p.item() for p in lambda_pred]
        lambda_pred = np.array(lambda_pred).reshape(-1, 1)
        lambda_pred = scaler.inverse_transform(lambda_pred).flatten().tolist()
        
        return idx, lambda_pred

    def cloze_grad(self, x, y, padding_mask):
        x = x.cuda()
        padding_mask = padding_mask.cuda() # [batch_size, max_len]
        y = y.cuda() # [batch_size, max_len]

        mask = (y < 20) & (padding_mask) # [batch_size, max_len]

        loss = 0
        correct = 0
        n = mask.sum() # total num of valid pos per batch

        if n > 0:
            logits, _ = self.clf(x, padding_mask)  # logits: [batch_size, max_len, 21]
            batch_size = logits.shape[0]
            max_len = logits.shape[1]


            logits = logits.view(-1, logits.shape[-1]) # [batch_size*max_len, 21]
            y = y.view(-1) # [batch_size*max_len]
            loss_seq = F.cross_entropy(logits, y, reduction = 'none') # [batch_size*max_len]
            loss_seq = loss_seq.view(batch_size, max_len) 
            loss_seq_mean = torch.stack([x[x!=0].mean() for x in loss_seq*mask]) 

            flat_mask = mask.view(-1).bool() # [batch_size*max_len]
            logits = logits[flat_mask] # [num_valid_tokens, 21]
            y = y[flat_mask] # [num_valid_tokens]
            loss_batch = F.cross_entropy(logits, y, reduction = 'mean') # scalar
            

            _, y_hat = torch.max(logits, 1) # [num_valid_tokens]
            correct = torch.sum((y == y_hat).float()).item()

        else: 
            raise Exception('No valid position.')
        return loss_batch, loss_seq_mean, correct, n

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
            x, y, padding_mask, idxs, t = next(iterator) # x: [batch_size, max_len, 21], y: [batch_size, max_len]

            idxs = np.array(train_indices)[idxs]  # original index
            lambdas = self.lambdas[idxs]
            slacks = self.slacks[idxs]
            lambdas = lambdas.detach().clone().to("cuda")
            slacks = slacks.detach().clone().to("cuda")
            self.flag[idxs] += 1

            optimizer.zero_grad() 

            loss_batch, loss_seq_mean, correct, b = self.cloze_grad(x, y, padding_mask) 

            self.n += b 
            delta = b*(loss_batch - self.lossCurrent)
            self.lossCurrent += delta/self.n
            delta = correct - b*self.accFinal
            self.accFinal += delta/self.n
            self.token += t

            constraint_violations = (loss_seq_mean - (self.epsilon+slacks.cuda())).nanmean().item() # usually 0 positive
            
            lagrangian = (loss_seq_mean*(1+lambdas)-lambdas*(self.epsilon+slacks.cuda())).nanmean() + 0.5*self.alpha*torch.linalg.norm(slacks)**2
            lagrangian.backward()

            avg_grad = self.compute_avg_abs_gradient(self.clf)
            for p in filter(lambda p: p.grad is not None, self.clf.parameters()): p.grad.data.clamp_(min=-.15, max=.15)
            if not np.isinf(self.opts['clip']):
                nn.utils.clip_grad_norm_(self.clf.encoder.parameters(), self.opts['clip'])

            optimizer.step()

            nan_mask = torch.isnan(loss_seq_mean)
            nan_idxs = torch.nonzero(nan_mask, as_tuple=True)
            loss_seq_mean[nan_idxs] = self.epsilon + slacks[nan_idxs] # skip nan when updating dual variables, replace epsilon with epsilon+slacks

            lambdas_current = lambdas # ?
            lambdas += self.lr_dual*(loss_seq_mean-(self.epsilon+slacks))
            slacks -= self.lr_slack*(0.5*self.alpha*slacks-lambdas_current) 
            lambdas.data.clamp_(min=0)
            slacks.data.clamp_(min=0)

    
            self.lambdas[idxs] = lambdas.detach().cpu()
            self.slacks[idxs] = slacks.detach().cpu()

            lambda_mean = self.lambdas[self.flag >= 1].mean().item() # log the mean of ALL lambdas with non-zero flags
            slack_mean = self.slacks[self.flag >= 1].mean().item()


            if i%self.opts['validate_every'] == 1 or self.opts['validate_every'] == 1:
                perplexity = self.validate_esm()
                self.clf.train()

            if i%self.opts['dual_lr_stepsize'] == 0:
                self.lr_dual = self.opts['dual_lr_gamma']*self.lr_dual
            
            scheduler.step() 
            
            wandb.log({'lambda_aver': lambda_mean, 'train loss': self.lossCurrent.item(), 'train acc': self.accFinal.item(), 'val perplexity': perplexity.item(), 
                        'avg abs grad': avg_grad, 'slack_aver': slack_mean, '# tokens': self.token, 'constraint_violations':constraint_violations,
                        'model_lr': optimizer.param_groups[0]["lr"]}) 

        self.lr_dual = self.opts['lr_dual']

        return self.lossCurrent.item(), self.accFinal.item()

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
                                            collate_fn=pad_seq)

        return loader

    def train(self):
        output_dir = self.opts['output'] + '/' + self.opts['name'] + '/' 
        os.makedirs(output_dir, exist_ok=True) 

        train_indices = np.arange(self.n_all)[self.idxs_base]
        train_loader = self.build_train_loader(train_indices)
        
        lossCurrent, accCurrent= self._PDCL(self.optimizer_clf, self.scheduler_clf, train_loader, train_indices)   

        # perplexity = self.validate()
        perplexity = self.validate_esm()
        print('val ppl', perplexity)

        if perplexity < self.perplexity_best:
            self.perplexity_best = perplexity
            self.epochs_no_improve = 0
            self.best_model = deepcopy(self.clf)
            output_path = output_dir + datetime.now().strftime('%Y%m%d_%H%M') +'.sav' 
            torch.save([self.best_model, self.reg], output_path) # save best model
        else:
            self.epochs_no_improve += 1

        print(f"training accuracy: {accCurrent:.2f} \tTraining loss: {lossCurrent:.2f}", flush=True)
        print('Done with training and validation:', time.strftime("%H:%M:%S", time.localtime()))
