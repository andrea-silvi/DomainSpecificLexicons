import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

from utils.glove_loader import load_glove_model

'''
We need [word_vector, target] -> label propagation -> [word, target]
'''


class SeedDataset(Dataset):
    def __init__(self, data, embeddings_path,  split='train'):

        self.split = split
        if self.split == 'train':
            self.words = list(data.keys())
            self.scores = list(data.values())
        else:
            self.words = list(data.keys())
        self.embeddings = load_glove_model(embeddings_path, data)

    def __getitem__(self, index):
        x = torch.tensor(self.embeddings[self.words[index]])
        if self.split == 'train':
            y = torch.tensor(self.scores[index])
            return x, y
        w = torch.tensor(self.words[index])
        return x, w

    def __len__(self):
        return len(self.words)

    def get_min_score(self):
        return min(self.scores)

    def get_max_score(self):
        return max(self.scores)

    def get_dictionary(self):
        if self.split == 'test':
            raise Exception("test dataset does not have a complete dictionary.")
        else:
            return {x: y for x, y in zip(self.words, self.scores)}
