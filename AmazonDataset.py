import argparse
import gzip
import json
from nltk import RegexpTokenizer
from utils.preprocessing import find_negations, whole_sentence_negation, find_complex_negations
import seaborn as sns
from utils.utils import timing_wrapper
import spacy
import time


def parse(path):
    g = gzip.open(path, 'rb')
    for l in g:
        yield json.loads(l)


# TODO : change negation type handling (not using booleans, rather enum type or something similar)
@timing_wrapper("dataset parsing")
def parse_dataset(dataset_name, negation='normal'):
    """
    Generate a numpy array with (review, score) from a gzip file.
    We throw away reviews with scores = 3 and we consider all ones below 3 as negative, and all
    ones above 3 as positive.
    """

    whole_negation = negation == 'whole'
    complex_negation = negation == 'complex'
    if complex_negation:
        nlp_parser = spacy.load("en_core_web_sm")
        for component in nlp_parser.pipe_names:
            if component != 'parser':
                nlp_parser.remove_pipe(component)

    reviews, scores = [], []
    tokenizer = RegexpTokenizer(r'\w+')
    new_time = time.time()
    for i, review in enumerate(parse(dataset_name)):
        try:
            if review["overall"] != 3.0:
                if whole_negation:
                    rev = whole_sentence_negation(review["reviewText"], tokenizer)
                elif complex_negation:
                    rev = find_complex_negations(review["reviewText"], nlp_parser)
                    if i % 250000 == 0 and i != 0:
                        print(f'{i} documents read in {time.time() - new_time}.')
                        new_time = time.time()
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
    complex_negation = negation == 'complex'
    if complex_negation:
        nlp_parser = spacy.load("en_core_web_sm")
        for component in ["tagger", "ner"]:
            nlp_parser.remove_pipe(component)
    reviews, scores = [], []
    tokenizer = RegexpTokenizer(r'\w+')

    for i, review in enumerate(parse(dataset_name)):
        if i == 2500000:
            break
        try:
            year_review = int(review["reviewTime"].split(",")[1][1:])
            if year_review in cluster:
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
        print(f'N. of reviews for year {k}: {v}.')
        plt = sns.histplot(data=reviewsPerYear)
        plt.savefig("Distribution_reviews_per_year.png")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, required=True, help='Path of the dataset.')
    args = parser.parse_args()
    distributionOverYears(args.dataset_name)
