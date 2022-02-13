import torch
from torch.utils.data import Dataset
import numpy as np
from utils_.glove_loader import load_glove_model


class SeedDataset(Dataset):
    def __init__(self, data, embeddings_path, split='train'):
        self.split = split
        if self.split == 'train':
            self.scores = np.array(list(data.values()), dtype=np.float32)
        self.words = list(data.keys())
        self.embeddings = load_glove_model(embeddings_path, data)

    def __getitem__(self, index):
        """
        If it is the training dataset, it returns (word_vector, score).
        If it is the test dataset, it returns (word_vector, word).
        """
        x = torch.tensor(self.embeddings[self.words[index]])
        if self.split == 'train':
            y = torch.tensor(self.scores[index])
            return x, y
        w = self.words[index]
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
