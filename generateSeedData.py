'''Generate seed data that will be used to train a network that can extract a domain dependent lexicon from it.
Seed data are in the form of (word, score).
'''

# import argparse
import os
from utils.utils import upload_args_from_json
import numpy as np
from AmazonDataset import parseDataset
from sklearn.feature_extraction.text import CountVectorizer
from liblinear.liblinearutil import predict, train, problem, parameter


def generate_bow(reviews):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(reviews)
    return X, vectorizer.vocabulary_


def train_linear_pred(X, y):
    w_negative = len(y[y == +1]) / len(y)
    w_positive = 1 - w_negative
    prob = problem(y, X)
    param = parameter(f'-w-1 {w_negative} -w+1 {w_positive}')
    m = train(prob, param)
    [W, _b] = m.get_decfun()
    predict(y, X, m)
    return W


def assign_word_labels(X, w, vocabulary, f_min):
    frequencies = X.sum(axis=0)
    ind = np.where(frequencies < f_min)
    filtered_vocabulary = {key: val for key, val in vocabulary.items() if
                           val not in ind and not key.startswith('negatedw')}
    for key, val in filtered_vocabulary.items():
        # build dataset for part 2
        continue


if __name__ == '__main__':
    '''opt = upload_args_from_json(os.path.join("parameters", "generate_seed_data.json"))
    if opt.isAmazon:
        data = parseDataset(opt)
    else:  # already a numpy array of shape (N, 2)
        pass
    processed_data = (data)
    X, y = generate_bow(processed_data)
    '''
