import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb
from torch.autograd import Variable
from torch.nn.functional import one_hot


class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes)
    
    def forward(self, x):
        logits = self.linear(x)  # Linear transformation
        probs = torch.softmax(logits, dim=1)  # Softmax for probabilities
        return probs

class LinearMapping(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(LinearMapping, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x)

class DTIPooling(nn.Module):
    def __init__(
        self, 
        drug_dim,
        target_dim,
        latent_dim):
        super(DTIPooling, self).__init__()
        self.drug_dim = drug_dim
        self.target_dim = target_dim
        self.latent_dim = latent_dim

        self.drug_projector = nn.Sequential(nn.Linear(self.drug_dim, self.latent_dim), nn.ReLU())
        nn.init.xavier_normal_(self.drug_projector[0].weight)
        self.target_projector = nn.Sequential(nn.Linear(self.target_dim, self.latent_dim), nn.ReLU())
        nn.init.xavier_normal_(self.target_projector[0].weight)

    def target_pooling(self, target):
        if len(target.shape) == 2:
            return target
        else:
            mask = (target != -1)
            aggregated_target = torch.sum(target * mask, dim=1) / torch.sum(mask, dim=1)
            return aggregated_target

    def forward(self, drug, target):
        drug_projection = self.drug_projector(drug)
        aggregated_target = self.target_pooling(target)
        target_projection = self.target_projector(aggregated_target.float())
        distance = nn.CosineSimilarity()(drug_projection, target_projection)
        return distance.squeeze()

class AAVNetwork(nn.Module):
    def __init__(self, input_dim):
        super(AAVNetwork, self).__init__()
        self.input_dim = input_dim
        self.fc1 = nn.Linear(input_dim, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x