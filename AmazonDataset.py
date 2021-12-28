import gzip, json
import os

import numpy as np
from nltk import RegexpTokenizer

from utils.utils import upload_args_from_json
from utils.preprocessing import find_negations


def parse(path):
  g = gzip.open(path, 'rb')
  for l in g:
    yield json.loads(l)

def parseDataset(dataset_name):
    '''
    Generate a numpy array with (review, score) from a gzip file.
    We throw away reviews with scores = 3 and we consider all ones below 3 as negative, and all
    ones above 3 as positive.
    '''
    reviews = []
    tokenizer = RegexpTokenizer(r'\w+')
    for review in parse(dataset_name):
        try:
            if review["overall"] != 3.0:
                r = []
                r.append(find_negations(review["reviewText"], tokenizer))
                r.append(-1 if review["overall"] < 3.0 else +1)
                review.append(r)
        except KeyError:
            continue

    return np.array(reviews)


if __name__ == '__main__':
    opt = upload_args_from_json(os.path.join("parameters", "AmazonDataset.json"))
    reviews = parseDataset(opt)
