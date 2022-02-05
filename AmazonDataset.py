import argparse
import gzip
import json
from nltk import RegexpTokenizer
from utils.preprocessing import find_negations, whole_sentence_negation
import seaborn as sns

def parse(path):
    g = gzip.open(path, 'rb')
    for l in g:
        yield json.loads(l)


# TODO : change negation type handling (not using booleans, rather enum type or something similar)
def parse_dataset(dataset_name, negation='normal'):
    """
    Generate a numpy array with (review, score) from a gzip file.
    We throw away reviews with scores = 3 and we consider all ones below 3 as negative, and all
    ones above 3 as positive.
    """
    
    whole_negation = negation=='whole'
    
    reviews, scores = [], []
    tokenizer = RegexpTokenizer(r'\w+')

    for review in parse(dataset_name):
        try:
            if review["overall"] != 3.0:
                if whole_negation:
                    rev = whole_sentence_negation(review["reviewText"], tokenizer)
                else:
                    rev = find_negations(review["reviewText"], tokenizer)
                
                score = -1 if review["overall"] < 3.0 else +1
                reviews.append(rev)
                scores.append(score)
        except KeyError:
            continue

    return reviews, scores


def parse_dataset_by_year(dataset_name, cluster, negation='normal'):
    """
    Generate a numpy array with (review, score) from a gzip file.
    We throw away reviews with scores = 3 and we consider all ones below 3 as negative, and all
    ones above 3 as positive.
    """
    whole_negation = negation == 'whole'
    reviews, scores = [], []
    tokenizer = RegexpTokenizer(r'\w+')

    for review in parse(dataset_name):
        try:
            year_review = int(review["reviewTime"].split(",")[1][1:])
            if year_review in cluster:
                if review["overall"] != 3.0:
                    if whole_negation:
                        rev = whole_sentence_negation(review["reviewText"], tokenizer)
                    else:
                        rev = find_negations(review["reviewText"], tokenizer)

                    score = -1 if review["overall"] < 3.0 else +1
                    reviews.append(rev)
                    scores.append(score)
        except KeyError:
            continue

    return reviews, scores

def distributionOverYears(dataset_name):
    """
    Get the dataset distribution of reviews over the years.
    """
    reviewsPerYear = {}
    for review in parse(dataset_name):
        try:
            year_review = int(review["reviewTime"].split(",")[1][1:])
            if year_review in reviewsPerYear:
                reviewsPerYear[year_review] += 1
            else:
                reviewsPerYear[year_review] = 1
        except KeyError:
            continue
    for k, v in reviewsPerYear.items():
        print(f'N. of reviews for year {k}: v.')
        plt = sns.histplot(data=reviewsPerYear)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, required=True, help='Path of the dataset.')
    args = parser.parse_args()
    distributionOverYears(args.dataset_name)
