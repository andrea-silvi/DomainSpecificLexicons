from sklearn.svm import LinearSVC
from dataset.SeedDataset import SeedDataset
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from utils_.utils import timing_wrapper


@timing_wrapper("BOW generation")
def generate_bow(reviews):
    """
    It creates a document term count matrix.
    """
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(reviews)
    return X, vectorizer.vocabulary_


@timing_wrapper("SVM training")
def train_linear_pred(X, y):
    """
    It trains an SVM classifier (with weighted loss) and obtain the linear predictor coefficients, to be used for seed
    data induction.
    """
    w_negative = len(y[y == +1]) / len(y)
    w_positive = 1 - w_negative
    clf = LinearSVC(random_state=0, tol=1e-4, class_weight={-1: w_negative, 1: w_positive}, fit_intercept=False)
    clf.fit(X, y)
    W = np.array(clf.coef_[0], dtype=np.float32)
    return W


def assign_word_labels(frequencies, w, vocabulary, f_min, embeddings_path, glove_words, weighing='normal'):
    """
    It creates dataset based on word - SVM scores association
    @ params frequencies : frequency of each word, at its index in the vocabulary
    @ params w : SVM weights
    @ params vocabulary : the features of input X of the SVM step
    @ params f_min : minimal frequency threshold (words with a frequency below this threshold will be removed)
    @ params embeddings_path : word vector file path
    @ params glove_words : glove words vocabulary
    @ params weighing : 'normal' : just use SVM scores. 'whole': use average of score and opposite of the negation if 
    its frequency is higher than f_min. We use this as (normal weight + negated word weight)/2
    """
    ind_to_be_removed = np.nonzero(frequencies < f_min)[0]
    whole_weighing = weighing == 'whole'
    if whole_weighing:
        offset = len('negatedw')
        """
        We create a dictionary of the negated features words and their scores that have a frequency higher than f_min 
        (thus they are NOT inside ind)
        """
        negated = {
            key[offset:]: w[val] for key, val in vocabulary.items()
            if (val not in ind_to_be_removed) and (key.lower().startswith('negatedw')) and key[offset:] in glove_words
        }
        """
        We then obtain the dictionary as a set of tuples (word: score) where score is either the linear predictor inside
        w or, if the negated word feature appeared more than f_min times in the input corpus, as the average of the 
        predictor of the positive feature and of the negative feature.
        """
        seed_data = {
            key: (w[val] if key not in negated else (w[val] + negated[key]) / 2)
            for key, val in vocabulary.items()
            if (val not in ind_to_be_removed)
               and (not key.lower().startswith('negatedw'))
               and key in glove_words
        }

    else:
        seed_data = {key: w[val] for key, val in vocabulary.items() if
                     (val not in ind_to_be_removed) and (not key.lower().startswith('negatedw')) and key in glove_words}

    return SeedDataset(seed_data, embeddings_path)


def get_frequencies(X):
    """
    Computes frequencies from count matrix X
    """
    frequencies = X.sum(axis=0)
    frequencies = np.asarray(frequencies)[0]
    return frequencies
