'''Generate seed data that will be used to train a network that can extract a domain dependent lexicon from it.
Seed data are in the form of (word, score).
'''

from sklearn.svm import LinearSVC

from SeedDataset import SeedDataset
from utils.utils import upload_args_from_json
import numpy as np
from AmazonDataset import parse_dataset
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize, MaxAbsScaler, StandardScaler
from sklearn.metrics import precision_recall_fscore_support
from utils.glove_loader import load_glove_words
from utils.utils import timing_wrapper



@timing_wrapper("BOW generation")
def generate_bow(reviews):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(reviews)
    return X, vectorizer.vocabulary_

@timing_wrapper("SVM training")
def train_linear_pred(X, y):
    w_negative = len(y[y == +1]) / len(y)
    w_positive = 1 - w_negative
    # TODO: check the result of this function
    clf = LinearSVC(random_state=0, tol=1e-4, class_weight={-1: w_negative, 1: w_positive}, fit_intercept=False)
    clf.fit(X, y)
    W = np.array(clf.coef_[0], dtype=np.float32)
    return W

# TODO : handle whole sentence negation
def assign_word_labels(frequencies, w, vocabulary, f_min, EMBEDDINGS_PATH, glove_words, weighing='normal'):
    """
    creates dataset based on word - SVM scores association
    @ params frequencies : frequency of each word, at its index in the vocabulary
    @ params w : SVM weights
    @ params f_min : minimal frequency threshold (words with a frequency below this threshold will be removed)*
    @ params EMBEDDINGS_PATH : glove embeddings file path
    @ params glove_words : glove words vocabulary
    @ params weighing : 'normal' : just use SVM scores. 'whole' : use average of score and opposite of the negation if possible
    """
    ind = np.nonzero(frequencies < f_min)[0]

    whole_weighing = weighing == 'whole'

    if whole_weighing:
        # if we use this
        # then as words ' weights (not negated), we use (normal weight + (- negated word weight))/2
        offset = len('negatedw')
        negated = {
            key[offset:] : w[val] for key, val in vocabulary.items()
            if (val not in ind) and (key.lower().startswith('negatedw')) and key[offset:] in glove_words
        }
        seed_data = {
            key: (w[val] if key not in negated else (w[val] + (-negated[key]))/2)
            for key, val in vocabulary.items()
            if (val not in ind)
            and (not key.lower().startswith('negatedw'))
            and key in glove_words
        }
    else:
        seed_data = {key: w[val] for key, val in vocabulary.items() if
                 (val not in ind) and (not key.lower().startswith('negatedw')) and key in glove_words}

    return SeedDataset(seed_data, EMBEDDINGS_PATH)


def get_frequencies(X):
    """
    for computing frequencies from count matrix X
    """
    frequencies = X.sum(axis=0)
    frequencies = np.asarray(frequencies)[0]
    return frequencies
