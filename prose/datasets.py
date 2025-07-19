"""
Copyright (C) Tristan Bepler - All Rights Reserved
Author: Tristan Bepler <tbepler@gmail.com>
"""

from __future__ import print_function, division

import sys
import os
import glob
import random
from PIL import Image
import numpy as np
import pandas as pd
import torch
from prose.alphabets import Uniprot21
import prose.scop as scop
import prose.fasta as fasta
from torch.utils.data import Dataset, DataLoader
import re
import pytorch_lightning as pl
from pathlib import Path
from prose.utils import target_collate_fn, collate_emb, drug_target_collate_fn
from torch.nn.utils.rnn import PackedSequence
import typing as T
from prose.ALLY.featurizer import Featurizer
from prose.utils import pad_seq, pad_seq_scl



class SCOPeDataset:
    def __init__(self, path='data/SCOPe/astral-scopedom-seqres-gd-sel-gs-bib-95-2.06.train.fa'
                , alphabet=Uniprot21(), augment=None):
        print('# loading SCOP sequences:', path, file=sys.stderr)

        self.augment = augment

        names, structs, sequences = self.load(path, alphabet)
            
        self.names = names
        self.x = [torch.from_numpy(x) for x in sequences]
        self.y = torch.from_numpy(structs)

        print('# loaded', len(self.x), 'sequences', file=sys.stderr)


    def load(self, path, alphabet):
        with open(path, 'rb') as f:
            names, structs, sequences = scop.parse_astral(f, encoder=alphabet)    
        # make sure no sequences of length 0 are included
        names_filtered = []
        structs_filtered = []
        sequences_filtered = []
        for i in range(len(sequences)):
            s = sequences[i]
            if len(s) > 0:
                names_filtered.append(names[i])
                structs_filtered.append(structs[i])
                sequences_filtered.append(s)
        names = names_filtered
        structs = np.stack(structs_filtered, 0)
        sequences = sequences_filtered

        return names, structs, sequences


    def __len__(self):
        return len(self.x)


    def __getitem__(self, i):
        x = self.x[i].long()
        if self.augment is not None:
            x = self.augment(x)
        return x, self.y[i]


class SCOPePairsDataset:
    def __init__(self, path='data/SCOPe/astral-scopedom-seqres-gd-sel-gs-bib-95-2.06.test.sampledpairs.txt'
                , alphabet=Uniprot21()):
        print('# loading SCOP sequence pairs:', path, file=sys.stderr)

        table = pd.read_csv(path, sep='\t')
        x0 = [x.encode('utf-8').upper() for x in table['sequence_A']]
        self.x0 = [torch.from_numpy(alphabet.encode(x)) for x in x0]
        x1 = [x.encode('utf-8').upper() for x in table['sequence_B']]
        self.x1 = [torch.from_numpy(alphabet.encode(x)) for x in x1]

        self.y = torch.from_numpy(table['similarity'].values).long()

        print('# loaded', len(self.x0), 'sequence pairs', file=sys.stderr)


    def __len__(self):
        return len(self.x0)


    def __getitem__(self, i):
        return self.x0[i].long(), self.x1[i].long(), self.y[i]


class ContactMapDataset:
    def __init__(self, path, root='data/SCOPe/pdbstyle-2.06'
                , k=1, min_length=0, max_length=0
                , alphabet=Uniprot21()
                , augment=None
                ):

        names, sequences, contact_maps = self.load(path, root, k=k)

        self.names = names
        self.x = [torch.from_numpy(alphabet.encode(s)) for s in sequences]
        self.y = contact_maps

        self.augment = augment

        self.min_length = min_length
        self.max_length = max_length

        self.fragment = False
        if self.min_length > 0 and self.max_length > 0:
            self.fragment = True

        print('# loaded', len(self.x), 'contact maps', file=sys.stderr)


    def load(self, path, root, k=1):
        print('# loading contact maps:', root, 'for sequences:', path, file=sys.stderr)

        with open(path, 'rb') as f:
            names,sequences = fasta.parse(f)

        # find all of the contact maps and index them by protein identifier
        cmap_paths = glob.glob(root + os.sep + '*' + os.sep + '*.png')
        cmap_index = {os.path.basename(path).split('.cmap-')[0] : path for path in cmap_paths}

        # match the sequences to the contact maps
        names_filtered = []
        sequences_filtered = []
        contact_maps = []
        for (name,seq) in zip(names, sequences):
            name = name.decode('utf-8')
            pid = name.split()[0]
            if pid not in cmap_index:
                # try changing first letter to 'd'
                # required for some SCOPe identifiers
                pid = 'd' + pid[1:]
            path = cmap_index[pid]
            # load the contact map image
            im = np.array(Image.open(path), copy=False)
            contacts = np.zeros(im.shape, dtype=np.float32)
            # set the positive, negative, and masked residue pairs
            contacts[im == 1] = -1
            contacts[im == 255] = 1
            # mask the matrix below the kth diagonal
            mask = np.tril_indices(contacts.shape[0], k=k)
            contacts[mask] = -1

            # filter out empty contact matrices
            if np.any(contacts > -1):
                contact_maps.append(torch.from_numpy(contacts))
                names_filtered.append(name)
                sequences_filtered.append(seq)

        return names_filtered, sequences_filtered, contact_maps

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        x = self.x[i]
        y = self.y[i]

        mi_length = self.min_length
        ma_length = self.max_length
        if self.fragment and len(x) > mi_length:
            l = np.random.randint(mi_length, ma_length+1)
            if len(x) > l:
                i = np.random.randint(len(x)-l+1)
                xl = x[i:i+l]
                yl = y[i:i+l,i:i+l]
            else:
                xl = x
                yl = y
            # make sure there are unmasked observations
            while torch.sum(yl >= 0) == 0:
                l = np.random.randint(mi_length, ma_length+1)
                if len(x) > l:
                    i = np.random.randint(len(x)-l+1)
                    xl = x[i:i+l]
                    yl = y[i:i+l,i:i+l]
            y = yl.contiguous()
            x = xl

        x = x.long()
        if self.augment is not None:
            x = self.augment(x)

        return x, y


class FastaDataset:
    def __init__(self, path, max_length=0, alphabet=Uniprot21(), debug=False):
        print('# loading fasta sequences:', path, file=sys.stderr)
        with open(path, 'rb') as f:
            if debug:
                count = 0
                names = []
                sequences = []
                for name,sequence in fasta.parse_stream(f):
                    if count > 10000:
                        break
                    names.append(name)
                    sequences.append(sequence)
                    count += 1
            else:
                names,sequences = fasta.parse(f)

        self.names = names
        self.x = [torch.from_numpy(alphabet.encode(s)) for s in sequences]
        self.max_length = max_length

        print('# loaded', len(self.x), 'sequences', file=sys.stderr)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        x = self.x[i]
        max_length = self.max_length
        if max_length > 0 and len(x) > max_length:
            # randomly sample a subsequence of length max_length
            j = random.randint(0, len(x) - max_length)
            x = x[j:j+max_length]
        return x#.long()


class ClozeDataset:
    def __init__(self, x, p, noise):
        self.x = x
        self.p = p
        self.noise = noise

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i): # frozen mask for validation, random seed depends on i, same for testing
        x = self.x[i]
        p = self.p
        n = len(self.noise) # number of tokens

        # create the random mask... i.e. which positions to infer 
        mask = torch.rand(len(x), device=x.device) # returns a tensor filled with random num of length of x
        mask = (mask < p).long() # we mask with probability p
        y = mask*x + (1-mask)*(n-1) # assign unmasked positions to (n-1) 

        # sample the masked positions from the noise distribution
        noise = torch.multinomial(self.noise, len(x), replacement=True) 
        x = (1-mask)*x + mask*noise

        return x, y, i

class ValPPLDataset:
    def __init__(self, x, noise):
        self.data = []

        for i in range(len(x)):
            x_i = x[i].long()
            x_orig = x_i.clone()
            mask = torch.rand(len(x_i))
            indicator = torch.full((len(mask),), -1, dtype=torch.long)

            indicator[mask >= 0.15] = -1
            indicator[mask < 0.8 * 0.15] = 1
            indicator[(0.8 * 0.15 <= mask) & (mask < 0.9 * 0.15)] = 2
            indicator[(mask >= 0.9 * 0.15) & (mask < 0.15)] = 0

            x_mod = x_i.clone()
            if (indicator == 1).sum() > 0:
                noise_sample = torch.multinomial(noise, (indicator == 1).sum().item(), replacement=True)
                x_mod[indicator == 1] = noise_sample
            if (indicator == 2).sum() > 0:
                rand_token = torch.randint(0, len(noise), size=((indicator == 2).sum().item(),), dtype=torch.long)
                x_mod[indicator == 2] = rand_token

            self.data.append((x_mod, x_orig, indicator))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]



class UnmaskedDataset: # for validation, original idx was kept
    def __init__(self, x, idx):
        self.x = x
        self.idx = idx

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.idx[i]

class SSPDataset:
    def __init__(self, path, alphabet=Uniprot21(), max_len=500):
        print('# loading fasta sequences:', path, file=sys.stderr)
        data = pd.read_csv(path)

        self.max_len = max_len
        self.x = [torch.from_numpy(alphabet.encode(s.replace(" ", '').encode('utf-8'))) for s in data['input']]
        dic = {'C': 0, 'E': 1, 'H': 2}
        y_b = [[dic[c] for c in y.replace(" ", "")] for y in data['dssp3']]
        self.y = [torch.from_numpy(np.array(y)) for y in y_b]
        print('# loaded', len(self.x), 'sequences', file=sys.stderr)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        x = self.x[i].long()
        y = self.y[i]

        # Truncate to max_len
        if x.size(0) > self.max_len:
            x = x[:self.max_len]
            y = y[:self.max_len]

        return x, y, i

class EmbDataset:
    def __init__(self, emb, y):
        self.emb = emb
        # y = [z.to(torch.int64) for z in y]
        self.y = y

    def __len__(self):
        return len(self.emb)

    def __getitem__(self, i):
        emb = self.emb[i]
        y = self.y[i]
        return emb, y

class SCLDataset(Dataset):
    def __init__(
        self,
        targets,
        labels,
        model,
        alphabet=Uniprot21(),
        max_len=500,
    ):
        self.targets = targets
        self.labels = labels
        self.model = model
        self.max_len = max_len

        self.targets = [
            torch.from_numpy(alphabet.encode(s.encode('utf-8'))[:max_len])  # Truncate to max_len
            for s in self.targets
        ]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i: int):
        target = self.targets[i]
        label = torch.tensor(self.labels.iloc[i])
        return target.long(), label, i


class SCLDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        model,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0,
        header=0,
        index_col=0,
        sep=",",
        max_len=500,
    ):
        self._loader_kwargs = {
            "batch_size": batch_size,
            "shuffle": shuffle,
            "num_workers": num_workers,
            "collate_fn": pad_seq_scl
        }

        self._csv_kwargs = {
            "header": header,
            "sep": sep,
        }

        self._data_dir = Path(data_dir)
        self._all_path = Path("balanced.csv")

        self._target_column = "sequence"
        self._label_column = "target"
        self._split_column = "set"
        self._val_column = "validation"

        self.model = model
        self.max_len = max_len

    def setup(self):
        df_all = pd.read_csv(self._data_dir / self._all_path, **self._csv_kwargs)

        self.df_train = df_all.loc[(df_all[self._split_column] == "train") & (df_all[self._val_column] != True)]
        self.df_val = df_all.loc[(df_all[self._split_column] == "train") & (df_all[self._val_column] == True)]
        self.df_test = df_all.loc[(df_all[self._split_column] == "test")]

        self._dataframes = [self.df_train, self.df_val, self.df_test]

        all_targets = pd.concat([i[self._target_column] for i in self._dataframes]).unique()
        unique_labels = self.df_train[self._label_column].unique()
        self.label_mapper = {x: i for i, x in enumerate(unique_labels)}

        self.data_train = SCLDataset(
            self.df_train[self._target_column],
            self.df_train[self._label_column].map(self.label_mapper),
            self.model,
            max_len=self.max_len,
        )

        self.data_val = SCLDataset(
            self.df_val[self._target_column],
            self.df_val[self._label_column].map(self.label_mapper),
            self.model,
            max_len=self.max_len,
        )

        self.data_test = SCLDataset(
            self.df_test[self._target_column],
            self.df_test[self._label_column].map(self.label_mapper),
            self.model,
            max_len=self.max_len,
        )


    def train_dataloader(self):
        return DataLoader(self.data_train, **self._loader_kwargs)

    def val_dataloader(self):
        return DataLoader(self.data_val, **self._loader_kwargs)

    def test_dataloader(self):
        return DataLoader(self.data_test, **self._loader_kwargs)


class BinaryDataset(Dataset):
    def __init__(
        self,
        drugs,
        targets,
        labels,
        drug_featurizer,
        target_featurizer,
        alphabet=Uniprot21(),
    ):
        self.drugs = drugs
        self.targets = targets
        self.labels = labels
        self.alphabet = alphabet

        self.drug_featurizer = drug_featurizer
        self.target_featurizer = target_featurizer

    def __len__(self):
        return len(self.drugs)

    def get_embedding(self, i):
        target = self.targets.iloc[i]
        target_encoded = torch.from_numpy(self.alphabet.encode(target.encode('utf-8'))).long().to('cuda')
        _, emb = self.target_featurizer(target_encoded) # [seq_len, 3093]
        emb = emb.mean(dim=0) # [3093], sequence-level
        return emb

    def __getitem__(self, i: int):
        drug = self.drug_featurizer(self.drugs.iloc[i])
        target = self.get_embedding(i)
        label = torch.tensor(self.labels.iloc[i])
        return drug, target.long(), label

class DTIDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        drug_featurizer: Featurizer,
        target_featurizer,
        device: torch.device = torch.device("cpu"),
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0,
        header=0,
        index_col=0,
        sep=",",
    ):
        self._loader_kwargs = {
            "batch_size": batch_size,
            "shuffle": shuffle,
            "num_workers": num_workers,
            "collate_fn": drug_target_collate_fn,
        }

        self._csv_kwargs = {
            "header": header,
            "index_col": index_col,
            "sep": sep,
        }

        self._device = device

        self._data_dir = Path(data_dir)
        self._train_path = Path("train.csv")
        self._val_path = Path("val.csv")
        self._test_path = Path("test.csv")

        self._drug_column = "SMILES"
        self._target_column = "Target Sequence"
        self._label_column = "Label"

        self.drug_featurizer = drug_featurizer
        self.target_featurizer = target_featurizer


    def setup(self, stage: T.Optional[str] = None):
        self.df_train = pd.read_csv(
            self._data_dir / self._train_path, **self._csv_kwargs
        )

        self.df_val = pd.read_csv(self._data_dir / self._val_path, **self._csv_kwargs)

        self.df_test = pd.read_csv(self._data_dir / self._test_path, **self._csv_kwargs)

        self._dataframes = [self.df_train, self.df_val, self.df_test]

        all_drugs = pd.concat([i[self._drug_column] for i in self._dataframes]).unique()
        all_targets = pd.concat(
            [i[self._target_column] for i in self._dataframes]
        ).unique()

        if self._device.type == "cuda":
            self.drug_featurizer.cuda(device=self._device)

        self.drug_featurizer.preload(all_drugs)

        self.data_train = BinaryDataset(
            self.df_train[self._drug_column],
            self.df_train[self._target_column],
            self.df_train[self._label_column],
            self.drug_featurizer,
            self.target_featurizer,
        )

        self.data_val = BinaryDataset(
            self.df_val[self._drug_column],
            self.df_val[self._target_column],
            self.df_val[self._label_column],
            self.drug_featurizer,
            self.target_featurizer,
        )

        self.data_test = BinaryDataset(
            self.df_test[self._drug_column],
            self.df_test[self._target_column],
            self.df_test[self._label_column],
            self.drug_featurizer,
            self.target_featurizer,
        )

    def train_dataloader(self):
        return DataLoader(self.data_train, **self._loader_kwargs)

    def val_dataloader(self):
        return DataLoader(self.data_val, **self._loader_kwargs)

    def test_dataloader(self):
        return DataLoader(self.data_test, **self._loader_kwargs)

class AAVDataset:
    def __init__(self, data, alphabet=Uniprot21(), max_len=500):
        data.reset_index(drop=True, inplace=True)
        self.x = [
            torch.from_numpy(alphabet.encode(s.encode('utf-8'))[:max_len])
            for s in data['full_aa_sequence']
        ]
        self.y = data['score']

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        x = self.x[i].long()
        y = self.y[i]
        return x, y, i


