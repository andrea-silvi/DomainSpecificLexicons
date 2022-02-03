import gzip
import json
from nltk import RegexpTokenizer
from utils.preprocessing import find_negations, find_complex_negations
from nltk.parse.stanford import StanfordDependencyParser

def parse(path):
    g = gzip.open(path, 'rb')
    for l in g:
        yield json.loads(l)


# TODO : change negation type handling (not using booleans, rather enum type or something similar)
def parse_dataset(dataset_name, complex_negations=False, whole_sentence_negation=False):
    """
    Generate a numpy array with (review, score) from a gzip file.
    We throw away reviews with scores = 3 and we consider all ones below 3 as negative, and all
    ones above 3 as positive.
    @ params : complex_negations and whole_sentence_negation cannot both be True
    """
    assert not (complex_negations and whole_sentence_negation), "complex_negations and whole_sentence_negation cannot both be True"
    reviews, scores = [], []
    tokenizer = RegexpTokenizer(r'\w+')

    if complex_negations:
        jar_path = '/content/stanford-corenlp-4.2.2/stanford-corenlp-4.2.2.jar'
        # Path to CoreNLP model jar
        models_jar_path = '/content/stanford-corenlp-4.2.2-models-english.jar'
        # Initialize StanfordDependency Parser from the path
        parser = StanfordDependencyParser(path_to_jar=jar_path, path_to_models_jar=models_jar_path)

    
    for review in parse(dataset_name):
        try:
            if review["overall"] != 3.0:
                if not complex_negations:
                    if whole_sentence_negation:
                        rev = whole_sentence_negation(review["reviewText"], tokenizer)
                    else:
                        rev = find_negations(review["reviewText"], tokenizer)
                else:
                    rev = find_complex_negations(review["reviewText"], tokenizer, parser, negations_list=['not', 'nor', 'never'])
                score = -1 if review["overall"] < 3.0 else +1
                reviews.append(rev)
                scores.append(score)
        except KeyError:
            continue

    return reviews, scores
