import torch
from torch.utils.data import DataLoader, Dataset
import argparse
from prose.datasets import SSPDataset
from prose.ALLY.prediction import SSPClassifier
import prose.ALLY.models as Classifiers
import numpy as np

import os
taskID=int(os.environ['SLURM_ARRAY_TASK_ID'])

def main():
    parser = argparse.ArgumentParser('Script for downstream task - secondary structure prediction')
    parser.add_argument('--path-train', default='/hpc/group/naderilab/eleanor/Efficient_PLM/data/CASP/Train_HHblits.csv', help='path to training dataset in csv format')
    parser.add_argument('--path-test', default='/hpc/group/naderilab/eleanor/Efficient_PLM/data/CASP/CASP12_HHblits.csv', help='path to testing dataset in csv format')
    parser.add_argument('--path-model', default='/hpc/group/naderilab/eleanor/prose_data/saved_models/_iter1000000.sav', help='path to pretrained model')
    parser.add_argument('--batch-size', type=int, default=100, help='batch size')
    parser.add_argument('-d', '--device', type=int, default=-2, help='compute device to use')
    parser.add_argument('-c', type=str, default='logistic', help='classifier used for this task')
    parser.add_argument('--nclass', type=int, default=3, help='number of class for classification task')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate of classifier')
    parser.add_argument('-n', type=int, default=10, help='number of epochs to train classifier')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='L2 regularization')
    parser.add_argument('--seed', default=1357, type=int, help='random seed for replication')

    args = parser.parse_args()
    args.seed = taskID
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    use_cuda = (args.device != -1) and torch.cuda.is_available()

    train = SSPDataset(args.path_train)
    test = SSPDataset(args.path_test)

    model = torch.load(args.path_model)


    opts = vars(args)

    ssp_cla = SSPClassifier(model, use_cuda, train, test, opts)
    ssp_cla.train()
    ssp_cla.predict()

if __name__ == '__main__':
    main()