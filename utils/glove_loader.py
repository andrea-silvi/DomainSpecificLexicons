import numpy as np
import logging
logger = logging.getLogger()


def load_glove_model(File, vocab=None):
    print("Loading Glove Model...")
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

def load_glove_words(File, ):
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


def load_glove_empty_dictionary(File, ):
    print("Loading Glove words...")
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