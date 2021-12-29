import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

from utils.glove_loader import load_glove_model

'''
We need [word_vector, target] -> label propagation -> [word, target]
'''


class SeedDataset(Dataset):
    def __init__(self, data, embeddings_path):
        self.words = list(data.keys())
        self.scores = list(data.values())
        self.embeddings = load_glove_model(embeddings_path, data)

    def __getitem__(self, index):
        x = torch.Tensor(self.embeddings[self.data[index]])
        y = torch.Tensor(self.scores[index])

        return x, y

    def __len__(self):
        return len(self.data)

    def get_min_score(self):
        return min(self.scores)

    def get_max_score(self):
        return max(self.scores)
