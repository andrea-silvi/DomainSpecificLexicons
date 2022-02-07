import nltk
from nltk import RegexpTokenizer
from utils.preprocessing import find_negations
nltk.download('punkt')
from convokit import Corpus, download


def parse_subreddit(subreddit, negation):
    corpus = Corpus(filename=download("subreddit-" + subreddit))
    comments, scores = [], []
    tokenizer = RegexpTokenizer(r'\w+')
    for utterance in corpus.iter_utterances():
        comments.append(find_negations(utterance.text, tokenizer))
        scores.append(1 if utterance.meta['score'] > 0 else -1)
    return comments, scores

