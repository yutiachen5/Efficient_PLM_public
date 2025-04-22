import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset

# Toy dataset
class ToyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, idx):
        return self.data[idx], idx  # Return data and index

    def __len__(self):
        return len(self.data)

# Custom sampler
class LargeWeightedRandomSampler(torch.utils.data.sampler.WeightedRandomSampler):
    """WeightedRandomSampler except allows for more than 2^24 samples to be sampled"""
    def __iter__(self):
        rand_tensor = np.random.choice(
            range(0, len(self.weights)),
            size=self.num_samples,
            p=self.weights.numpy() / torch.sum(self.weights).numpy(),
            replace=self.replacement
        )
        rand_tensor = torch.from_numpy(rand_tensor)
        return iter(rand_tensor.tolist())

# Example data
data = ['A', 'B', 'C', 'D', 'E']
dataset = ToyDataset(data)

# Define weights
weights = torch.tensor([0.4, 0.1, 0.1, 0.1, 0.3])  # Higher probability for 'A' and 'E'

# Instantiate custom sampler
sampler = LargeWeightedRandomSampler(weights, num_samples=6, replace=False)

# DataLoader using the custom sampler
loader_with_custom_sampler = DataLoader(dataset, batch_size=2, sampler=sampler)

for batch_data, batch_indices in loader_with_custom_sampler:
    print("Batch data:", batch_data)
    print("Batch indices:", batch_indices)
