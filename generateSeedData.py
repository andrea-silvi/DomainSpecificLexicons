'''Generate seed data that will be used to train a network that can extract a domain dependent lexicon from it.
Seed data are in the form of (word, score).
'''

# import argparse
import os
from sklearnex import patch_sklearn
patch_sklearn()
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

from SeedDataset import SeedDataset
from utils.utils import upload_args_from_json
import numpy as np
from AmazonDataset import parse_dataset
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize, MaxAbsScaler
from liblinear.liblinearutil import predict, train, problem, parameter
from sklearn.metrics import precision_recall_fscore_support
from utils.glove_loader import load_glove_words




def generate_bow(reviews):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(reviews)
    return X, vectorizer.vocabulary_


def train_linear_pred(X, y, print_overfitting=False):
    w_negative = len(y[y == +1]) / len(y)
    w_positive = 1 - w_negative

    # we first normalize X
    # TODO: check the result of this function
    # X = normalize(X, norm='l1', copy=False)
    # scaler = MinMaxScaler() works not with sparse
    clf = make_pipeline(MaxAbsScaler(), LinearSVC(random_state=0, tol=1e-5,
                                                  class_weight={-1: w_negative, 1: w_positive}))
    clf.fit(X, y)
    W = np.array(clf.named_steps['linearsvc'].coef_[0], dtype=np.float32)
    return W


def assign_word_labels(frequencies, w, vocabulary, f_min, EMBEDDINGS_PATH, glove_words):
    ind = np.nonzero(frequencies < f_min)[0]
    seed_data = {key: w[val] for key, val in vocabulary.items() if
                 (val not in ind) and (not key.startswith('negatedw')) and key in glove_words}
    return SeedDataset(seed_data, EMBEDDINGS_PATH)


def get_frequencies(X):
    """
    for computing frequencies from count matrix X
    """
    frequencies = X.sum(axis=0)
    frequencies = np.asarray(frequencies)[0]
    return frequencies
