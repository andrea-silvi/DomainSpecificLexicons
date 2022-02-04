import gzip
import json
from nltk import RegexpTokenizer
from utils.preprocessing import find_negations, find_complex_negations
from nltk.parse.stanford import StanfordDependencyParser
import wget
import zipfile

def parse(path):
    g = gzip.open(path, 'rb')
    for l in g:
        yield json.loads(l)


def instanciate_parser():
    wget.download('https://nlp.stanford.edu/software/stanford-corenlp-4.2.2.zip')
    wget.download('https://nlp.stanford.edu/software/stanford-corenlp-4.2.2-models-english.jar')

    with zipfile.ZipFile('/content/stanford-corenlp-4.2.2.zip', 'r') as zip_ref:
        zip_ref.extractall('/content/stanford-corenlp-4.2.2')
    jar_path = '/content/stanford-corenlp-4.2.2/stanford-corenlp-4.2.2.jar'
    # Path to CoreNLP model jar
    models_jar_path = '/content/stanford-corenlp-4.2.2-models-english.jar'
    # Initialize StanfordDependency Parser from the path
    parser = StanfordDependencyParser(path_to_jar=jar_path, path_to_models_jar=models_jar_path)
    return parser

def parse_dataset(dataset_name, complex_negations=False):
    """
    Generate a numpy array with (review, score) from a gzip file.
    We throw away reviews with scores = 3 and we consider all ones below 3 as negative, and all
    ones above 3 as positive.
    """
    reviews, scores = [], []
    tokenizer = RegexpTokenizer(r'\w+')
    if complex_negations:
        parser = instanciate_parser()

    for review in parse(dataset_name):
        try:
            if review["overall"] != 3.0:
                if not complex_negations:
                    rev = find_negations(review["reviewText"], tokenizer)
                else:
                    rev = find_complex_negations(review["reviewText"], tokenizer, parser, negations_list=['not', 'nor', 'never'])
                score = -1 if review["overall"] < 3.0 else +1
                reviews.append(rev)
                scores.append(score)
        except KeyError:
            continue

    return reviews, scores
