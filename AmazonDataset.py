import gzip
import json
from nltk import RegexpTokenizer
from utils.preprocessing import find_negations


def parse(path):
    g = gzip.open(path, 'rb')
    for l in g:
        yield json.loads(l)


def parse_dataset(dataset_name):
    """
    Generate a numpy array with (review, score) from a gzip file.
    We throw away reviews with scores = 3 and we consider all ones below 3 as negative, and all
    ones above 3 as positive.
    """
    reviews, scores = [], []
    tokenizer = RegexpTokenizer(r'\w+')
    for review in parse(dataset_name):
        try:
            if review["overall"] != 3.0:
                rev = find_negations(review["reviewText"], tokenizer)
                score = -1 if review["overall"] < 3.0 else +1
                reviews.append(rev)
                scores.append(score)
        except KeyError:
            continue

    return reviews, scores
