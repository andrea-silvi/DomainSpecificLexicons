'''Generate seed data in order to train a network that can extract a domain dependent lexicon from them.
Seed data are in the form of (word, score).
Pass a numpy array of structure (comment, score)'''

# import argparse
import os
from utils.utils import upload_args_from_json
import numpy as np
from AmazonDataset import parseDataset
from sklearn.feature_extraction.text import CountVectorizer


def generate_bow(data):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(data[:, 0])
    y = data[:, 1]
    return X, y


if __name__ == '__main__':
    opt = upload_args_from_json(os.path.join("parameters", "generate_seed_data.json"))
    if opt.isAmazon:
        data = parseDataset(opt)
    else:  # already a numpy array of shape (N, 2)
        pass
    processed_data = (data)
    X, y = generate_bow(processed_data)

'this stuff is !notgood'
