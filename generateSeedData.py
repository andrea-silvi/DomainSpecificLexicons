'''Generate seed data that will be used to train a network that can extract a domain dependent lexicon from it.
Seed data are in the form of (word, score).
'''

from sklearn.svm import LinearSVC

from SeedDataset import SeedDataset
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer




def generate_bow(reviews):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(reviews)
    return X, vectorizer.vocabulary_


def train_linear_pred(X, y):
    w_negative = len(y[y == +1]) / len(y)
    w_positive = 1 - w_negative
    # TODO: check the result of this function
    clf = LinearSVC(random_state=0, tol=1e-5, class_weight={-1: w_negative, 1: w_positive}, fit_intercept=False)
    clf.fit(X, y)
    W = np.array(clf.coef_[0], dtype=np.float32)
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
