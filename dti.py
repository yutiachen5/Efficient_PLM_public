# adapted from: https://github.com/navid-naderi/PLM_SWE/blob/main/run_scl.py
from time import time
import numpy as np
import pandas as pd
import torch
import argparse
from prose.datasets import DTIDataModule
from prose.ALLY.featurizer import get_featurizer
from prose.ALLY.prediction import DTIClassifier

def main():
    parser = argparse.ArgumentParser('Script for downstream task - Subcelluar Localization Prediction')
    parser.add_argument('--path-model', default='/hpc/group/naderilab/eleanor/prose_data/saved_models/_iter1000000.sav', help='path to pretrained model')
    parser.add_argument('--data-dir', default='/hpc/group/naderilab/eleanor/Efficient_PLM/data/DAVIS', help='path for input data')
    parser.add_argument('-n', type=int, default=10, help='number of epochs to train classifier')
    parser.add_argument('--batch-size', type=int, default=100, help='batch size')
    parser.add_argument('--nclass', type=int, default=10, help='number of class for classification task')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate of classifier')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers for PyTorch DataLoader')
    parser.add_argument('-d', '--device', type=int, default=0, help='compute device to use')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='L2 regularization')
    parser.add_argument('--drug-featurizer', type=str, default='MorganFeaturizer', help='drug featurizer')
    parser.add_argument('--save-dir', default='/hpc/group/naderilab/eleanor/Efficient_PLM/data/datasets/DAVIS', help='path for saving features')
    parser.add_argument('--seed', default=1357, type=int, help='random seed for replication')
    parser.add_argument('--latent-dim', default=1024, type=int, help='dimension of shared co-embedding space')

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model = torch.load(args.path_model)
    drug_featurizer = get_featurizer(args.drug_featurizer, save_dir=args.save_dir)

    datamodule = DTIDataModule(
        data_dir=args.data_dir,
        drug_featurizer=drug_featurizer,
        target_featurizer=model,
        device=torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    datamodule.setup()
    drug_dim = drug_featurizer.shape

    opts = vars(args)

    dti_cla = DTIClassifier(model, datamodule, drug_dim, opts)
    dti_cla.train()
    dti_cla.predict()

if __name__ == '__main__':
    main()