import numpy as np
import logging
from utils_.utils import timing_wrapper

logger = logging.getLogger()


@timing_wrapper("Glove model loading")
def load_glove_model(File, vocab=None):
    """
    Opens the given GloVe word vector file and saves the word vectors.
    """
    print("Loading Glove word vectors...")
    glove_model = {}
    with open(File, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                split_line = line.split()
                word = split_line[0]
                if vocab == None or word in vocab:
                    embedding = np.array(split_line[1:], dtype=np.float32)
                    if len(embedding) == 300:
                        glove_model[word] = embedding
            except ValueError:
                continue
    return glove_model


def check_cast_to_float(s):
    try:
        x = float(s)
        return True
    except ValueError:
        return False


@timing_wrapper("Glove words loading")
def load_glove_words(File):
    """
    Returns only the GloVe vocabulary (WITHOUT word vectors) as a set.
    """
    print("Loading Glove words...")
    glove_words = set()
    with open(File, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                split_line = line.split()
                word = split_line[0]
                try:
                    t = float(split_line[1])
                    if not check_cast_to_float(word):
                        glove_words.add(word)
                    else:
                        logger.debug("Invalid word parsed in load_glove_words : word was actually a float")

                except ValueError:
                    continue
            except ValueError:
                continue

    return glove_words


def load_glove_empty_dictionary(File):
    """
    Returns only the GloVe vocabulary (WITHOUT word vectors) as a dictionary of (word: [empty list]).
    """
    print("Loading Glove words dictionary...")
    glove_words = {}
    with open(File, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                split_line = line.split()
                word = split_line[0]
                try:
                    t = float(split_line[1])
                    if not check_cast_to_float(word):
                        glove_words[word] = []
                    else:
                        logger.debug("Invalid word parsed in load_glove_words : word was actually a float")
                except ValueError:
                    continue
            except ValueError:
                continue
    return glove_words
