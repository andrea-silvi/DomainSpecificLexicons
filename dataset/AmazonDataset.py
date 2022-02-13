import argparse
import gzip
import json
from nltk import RegexpTokenizer
from utils_.preprocessing import find_negations, whole_sentence_negation, find_complex_negations
from utils_.utils import timing_wrapper
import spacy
import time


def parse(path):
    g = gzip.open(path, 'rb')
    for l in g:
        yield json.loads(l)


@timing_wrapper("dataset parsing")
def parse_dataset(dataset_name, negation='normal'):
    """
    Generate two lists of reviews and relative scores from a gzip file.
    We throw away reviews with scores = 3 and we consider all ones below 3 as negative, and all
    ones above 3 as positive.
    We process each review based on the chosen negation type.
    """

    whole_negation = negation == 'whole'
    complex_negation = negation == 'complex'
    if complex_negation:
        nlp_parser = spacy.load("en_core_web_sm")
        nlp_parser = remove_pipeline(nlp_parser)
    reviews, scores = [], []
    tokenizer = RegexpTokenizer(r'\w+')
    for i, review in enumerate(parse(dataset_name)):
        try:
            if review["overall"] != 3.0:
                if whole_negation:
                    rev = whole_sentence_negation(review["reviewText"], tokenizer)
                elif complex_negation:
                    rev = find_complex_negations(review["reviewText"], nlp_parser)
                else:
                    rev = find_negations(review["reviewText"], tokenizer)
                score = -1 if review["overall"] < 3.0 else +1
                reviews.append(rev)
                scores.append(score)
        except KeyError:
            continue
    return reviews, scores


def remove_pipeline(nlp_parser):
    """
    Removes all of spacy modules that we do not use except the Dependency Parser.
    """
    for component in nlp_parser.pipe_names:
        if component != 'parser':
            nlp_parser.remove_pipe(component)
    return nlp_parser


@timing_wrapper("dataset parsing")
def parse_dataset_by_year(dataset_name, cluster, negation='normal'):
    """
    Generate two lists of reviews and scores from a gzip file.
    We individuate if the review comes from the range of years we are interested in. Then we throw away reviews with
    scores = 3 and we consider all ones below 3 as negative, and all ones above 3 as positive.
    We process each review based on the chosen negation type.
    In order to limit the input corpus size, we stop at 2.5 million reviews.
    """
    whole_negation = negation == 'whole'
    complex_negation = negation == 'complex'
    if complex_negation:
        nlp_parser = spacy.load("en_core_web_sm")
        nlp_parser = remove_pipeline(nlp_parser)
    reviews, scores = [], []
    tokenizer = RegexpTokenizer(r'\w+')
    counter_reviews = 0
    for i, review in enumerate(parse(dataset_name)):
        if counter_reviews == 2500000:
            break
        try:
            year_review = int(review["reviewTime"].split(",")[1][1:])
            if year_review in cluster:
                if review["overall"] != 3.0:
                    counter_reviews += 1
                    if whole_negation:
                        rev = whole_sentence_negation(review["reviewText"], tokenizer)
                    elif complex_negation:
                        rev = find_complex_negations(review["reviewText"], nlp_parser)
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
    reviews_per_year = {}
    for review in parse(dataset_name):
        try:
            year_review = int(review["reviewTime"].split(",")[1][1:])
            if year_review in reviews_per_year:
                reviews_per_year[year_review] += 1
            else:
                reviews_per_year[year_review] = 1
        except KeyError:
            continue
    for k, v in reviews_per_year.items():
        print(f'N. of reviews for year {k}: {v}.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, required=True, help='Path of the dataset.')
    args = parser.parse_args()
    distributionOverYears(args.dataset_name)
