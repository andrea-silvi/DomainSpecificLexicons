import nltk
from nltk import RegexpTokenizer
from utils_.preprocessing import find_negations
nltk.download('punkt')
from convokit import Corpus, download


def parse_subreddit(subreddit):
    """
    Creates the ConvoKit Corpus object of the chosen subreddit and parse the comments and posts that are not neutral
    (considered as a score of 1).
    """
    corpus = Corpus(filename=download("subreddit-" + subreddit))
    comments, scores = [], []
    tokenizer = RegexpTokenizer(r'\w+')
    for utterance in corpus.iter_utterances():
        comments.append(find_negations(utterance.text, tokenizer))
        scores.append(1 if utterance.meta['score'] > 0 else -1)
    return comments, scores

