from time import time
import numpy as np
import pandas as pd
import torch
import argparse
from sklearn.model_selection import train_test_split
from prose.datasets import AAVDataset
from prose.ALLY.prediction_tf import AAVRegressor

def main():
    parser = argparse.ArgumentParser('Script for downstream task - adeno-associated virus 2 score prediction')
    parser.add_argument('--path-model', default='/hpc/group/naderilab/eleanor/Efficient_PLM/saved_models/save_emb_d_1024/20250402_1556.sav', help='path to pretrained model')
    parser.add_argument('--data-dir', default='/hpc/group/naderilab/eleanor/Efficient_PLM/data/AAV/full_data.csv', help='path for input data')
    parser.add_argument('-n', type=int, default=10, help='number of epochs to train regressor')
    parser.add_argument('--batch-size', type=int, default=100, help='batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate of regressor')
    parser.add_argument('-d', '--device', type=int, default=0, help='compute device to use')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='L2 regularization')
    parser.add_argument('--seed', default=1357, type=int, help='random seed for replication')
    parser.add_argument('--patience', default=2, type=int, help='early stop parameters')

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    model = torch.load(args.path_model)[0]

    data = pd.read_csv(args.data_dir)
    train, temp_df = train_test_split(data, test_size=0.3, random_state=args.seed)
    validation, test = train_test_split(temp_df, test_size=0.5, random_state=args.seed)

    train = AAVDataset(train)
    validation = AAVDataset(validation)
    test = AAVDataset(test)

    opts = vars(args)
    use_cuda = (args.device != -1) and torch.cuda.is_available()

    avv_reg = AAVRegressor(model, use_cuda, train, validation, test, opts)
    avv_reg.train()
    avv_reg.predict()

if __name__ == '__main__':
    main()