import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

'''
We need [word_vector, target] -> label propagation -> [word, target]'''


class SeedDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = torch.Tensor(targets)

    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]

        return x, y

    def __len__(self):
        return len(self.data)
