import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset



class lambdanet(nn.Module):
    
    def __init__(self, input_dim):
        super(lambdanet, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(),

            nn.Linear(256, 16),
            nn.LeakyReLU(),

            nn.Linear(16, 1),
            nn.Softplus()  
        )


    def forward(self, x):
        return self.layers(x) 

class lambdaset(Dataset):
    def __init__(self, X_train, X_test, y_train, y_test, train=True):

        if train:
            self.x_data, self.y_data = X_train, y_train
        else:
            self.x_data, self.y_data = X_test, y_test
    
    def __getitem__(self, i):
        return self.x_data[i], self.y_data[i], i

    def __len__(self):
        return self.y_data.shape[0]
