import numpy as np
import pdb
import torch
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import os
import os
from torchvision import transforms as T

def get_dataset(name, path, redund = 0):
    if name == 'MNIST':
        return get_MNIST(path)
    elif name == 'STL10':
        return get_STL10(path, redund)
    elif name == 'SVHN':
        return get_SVHN(path, redund)
    elif name == 'CIFAR10':
        return get_CIFAR10(path)
    elif name == 'TINY_IMAGENET':
        return get_TINY_IMAGENET(path)
    else:
        assert 0, "Wrong Dataset Name."


def get_handler():
    return DataHandler


class DataHandler(Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        return x, y, index

    def __len__(self):
        return len(self.X)

