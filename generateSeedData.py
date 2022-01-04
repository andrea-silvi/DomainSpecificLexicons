'''Generate seed data that will be used to train a network that can extract a domain dependent lexicon from it.
Seed data are in the form of (word, score).
'''

# import argparse
import os
from SeedDataset import SeedDataset
from utils.utils import upload_args_from_json
import numpy as np
from AmazonDataset import parse_dataset
from sklearn.feature_extraction.text import CountVectorizer
from liblinear.liblinearutil import predict, train, problem, parameter
from sklearn.metrics import precision_recall_fscore_support

EMBEDDINGS_PATH = '/content/drive/MyDrive/glove.840B.300d.txt'


def generate_bow(reviews):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(reviews)
    return X, vectorizer.vocabulary_


def train_linear_pred(X, y, print_overfitting=False):
    w_negative = len(y[y == +1]) / len(y)
    w_positive = 1 - w_negative
    prob = problem(y, X)
    param = parameter(f'-w-1 {w_negative} -w+1 {w_positive}')
    m = train(prob, param)
    [W, _b] = m.get_decfun()
    if print_overfitting:
        p_label, p_acc, p_val = predict(y, X, m)
        print(precision_recall_fscore_support(y, p_label))
    return W


def assign_word_labels(frequencies, w, vocabulary, f_min):
    ind = np.nonzero(frequencies < f_min)[0]
    seed_data = {key: w[val] for key, val in vocabulary.items() if (val not in ind) and (not key.startswith('negatedw'))}
    non_seed_data = {key: 0 for key, val in vocabulary.items() if (val in ind) and (not key.startswith('negatedw'))}
    return SeedDataset(seed_data, EMBEDDINGS_PATH),  SeedDataset(non_seed_data, EMBEDDINGS_PATH, split='test')

def get_frequencies(X):
    """
    for computing frequencies from count matrix X
    """
    frequencies = X.sum(axis=0)
    frequencies = np.asarray(frequencies)[0]
    return frequencies