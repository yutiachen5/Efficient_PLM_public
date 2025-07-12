from __future__ import print_function,division

import numpy as np
from pathlib import Path
import torch
import torch.utils.data
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import typing as T


def get_project_root() -> Path:
    return Path(__file__).parent.parent


def pack_sequences(X, order=None):
    #X = [x.squeeze(0) for x in X]

    n = len(X)
    lengths = np.array([len(x) for x in X])
    if order is None:
        order = np.argsort(lengths)[::-1]
        order = np.ascontiguousarray(order)
    m = max(len(x) for x in X)

    if len(X[0].size()) > 1:
        d = X[0].size(1)
        X_block = X[0].new(n,m,d).zero_()
        
        for i in range(n):
            j = order[i]
            x = X[j]
            X_block[i,:len(x),:] = x
    else:  
        X_block = X[0].new(n,m).zero_()
        
        for i in range(n):
            j = order[i]
            x = X[j]
            X_block[i,:len(x)] = x
        
    lengths = lengths[order]
    X = pack_padded_sequence(X_block, lengths, batch_first=True)

    return X, order 


def unpack_sequences(X, order):
    X,lengths = pad_packed_sequence(X, batch_first=True)
    X_block = [None]*len(order)
    for i in range(len(order)):
        j = order[i]
        X_block[j] = X[i,:lengths[i]]
    return X_block


def infinite_iterator(it):
    while True:
        for x in it:
            yield x


class LargeWeightedRandomSampler(torch.utils.data.sampler.WeightedRandomSampler):
    """WeightedRandomSampler except allows for more than 2^24 samples to be sampled"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __iter__(self):
        rand_tensor = np.random.choice(range(0, len(self.weights)),
                                       size=self.num_samples,
                                       p=self.weights.numpy() / torch.sum(self.weights).numpy(),
                                       replace=self.replacement)
        rand_tensor = torch.from_numpy(rand_tensor)
        return iter(rand_tensor.tolist())


def collate_seq2seq(args):
    x, y, i = zip(*args)
    x = list(x)
    y = list(y)
    i = list(i)

    x, order = pack_sequences(x)
    y,_ = pack_sequences(y, order=order)

    return x, y, i, order

def collate_seq2seq_unmasked(args):
    x, i = zip(*args)
    x = list(x)
    i = list(i)

    x, order = pack_sequences(x)

    return x, i, order

def collate_lists(args):
    x = [a[0] for a in args]
    y = [a[1] for a in args]
    return x, y

def collate_emb(args):
    x, y, i = zip(*args)
    x = list(x)
    y = list(y)
    i = list(i)

    x, order = pack_sequences(x)

    return x, y, i, order

def target_collate_fn(args: tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
    """
    Collate function for PyTorch data loader -- turn a batch of triplets into a triplet of batches

    If target embeddings are not all the same length, it will zero pad them
    This is to account for differences in length from FoldSeek embeddings

    :param args: Batch of training samples with molecule, protein, and affinity
    :type args: Iterable[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
    :return: Create a batch of examples
    :rtype: T.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    """
    # d_emb = [a[0] for a in args]
    t_emb = [a[0] for a in args]
    labs = [a[1] for a in args]

    # drugs = torch.stack(d_emb, 0)
    targets = pad_sequence(t_emb, batch_first=True, padding_value=-1)
    labels = torch.stack(labs, 0)

    return targets, labels

def drug_target_collate_fn(args: T.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
    """
    Collate function for PyTorch data loader -- turn a batch of triplets into a triplet of batches

    If target embeddings are not all the same length, it will zero pad them
    This is to account for differences in length from FoldSeek embeddings

    :param args: Batch of training samples with molecule, protein, and affinity
    :type args: Iterable[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
    :return: Create a batch of examples
    :rtype: T.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    """
    d_emb = [a[0] for a in args]
    t_emb = [a[1] for a in args]
    labs = [a[2] for a in args]

    drugs = torch.stack(d_emb, 0)
    targets = pad_sequence(t_emb, batch_first=True, padding_value=-1)
    labels = torch.stack(labs, 0)
    return drugs, targets, labels

def pad_seq(args):
    x, y, i = zip(*args)
    x = list(x)
    y = list(y)
    tokens = [len(seq) for seq in x]
    padded_x = pad_sequence(x, batch_first=True, padding_value=0)
    padded_y = pad_sequence(y, batch_first=True, padding_value=0)
    padding_mask = torch.tensor([[1]*l + [0]*(padded_x.shape[1] - l) for l in tokens], dtype=torch.bool) # 1:unpadded, 0: padded
    one_hot_x = F.one_hot(padded_x.to(torch.int64), num_classes=21).float()

    return one_hot_x, padded_y, padding_mask, list(i), sum(tokens)

def pad_seq_scl(args):
    x, y, i = zip(*args)
    x = list(x)
    y = list(y)

    tokens = [len(seq) for seq in x]
    padded_x = pad_sequence(x, batch_first=True, padding_value=0)
    padding_mask = torch.tensor([[1]*l + [0]*(padded_x.shape[1] - l) for l in tokens], dtype=torch.bool) # 1:unpadded, 0: padded
    one_hot_x = F.one_hot(padded_x.to(torch.int64), num_classes=21).float()
    y = [torch.tensor(label, dtype=torch.float32) for label in y]

    return one_hot_x, torch.stack(y), padding_mask, list(i), sum(tokens)

def pad_seq_val(args):
    x, x_orig, indicator = zip(*args)
    x = list(x)
    x_orig = list(x_orig)
    indicator = list(indicator)

    tokens = [len(seq) for seq in x]
    padded_x = pad_sequence(x, batch_first=True, padding_value=0)
    padded_x_orig = pad_sequence(x_orig, batch_first=True, padding_value=0) # same shape, share the same padding mask
    masking_indicator = pad_sequence(indicator, batch_first=True, padding_value=-1)

    padding_mask = torch.tensor([[1]*l + [0]*(padded_x.shape[1] - l) for l in tokens], dtype=torch.bool) 
    one_hot_x = F.one_hot(padded_x.to(torch.int64), num_classes=21).float()

    return one_hot_x, padded_x_orig, padding_mask, masking_indicator

def pad_seq_emb(args):
    x, i = zip(*args)
    x = list(x)
    i = list(i)

    tokens = [len(seq) for seq in x]
    padded_x = pad_sequence(x, batch_first=True, padding_value=0)
    padding_mask = torch.tensor([[1]*l + [0]*(padded_x.shape[1] - l) for l in tokens], dtype=torch.bool) 
    one_hot_x = F.one_hot(padded_x.to(torch.int64), num_classes=21).float()

    return one_hot_x, padding_mask, i


class AllPairsDataset(torch.utils.data.Dataset):
    def __init__(self, X, Y, augment=None):
        self.X = X
        self.Y = Y
        self.augment = augment

    def __len__(self):
        return len(self.X)**2

    def __getitem__(self, k):
        n = len(self.X)
        i = k//n
        j = k%n

        x0 = self.X[i].long()
        x1 = self.X[j].long()
        if self.augment is not None:
            x0 = self.augment(x0)
            x1 = self.augment(x1)

        y = self.Y[i,j]
        #y = torch.cumprod((self.Y[i] == self.Y[j]).long(), 0).sum()

        return x0, x1, y


def collate_paired_sequences(args):
    x0 = [a[0] for a in args]
    x1 = [a[1] for a in args]
    y = [a[2] for a in args]
    return x0, x1, torch.stack(y, 0)


class MultinomialResample:
    def __init__(self, trans, p):
        self.p = (1-p)*torch.eye(trans.size(0)).to(trans.device) + p*trans

    def __call__(self, x):
        p = self.p[x] # get distribution for each x
        return torch.multinomial(p, 1).view(-1) # sample from distribution
