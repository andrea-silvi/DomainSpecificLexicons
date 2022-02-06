import numpy as np
from nltk import RegexpTokenizer
from sklearn.metrics import precision_recall_fscore_support
from testDatasets import parseIMDBDataset, parseGameStopDataset


# TODO : instead of 0 for unknown words, use average sentiment score ?
def calculate_sentiment(tokens, lexicon):
    pred = 0
    for w in tokens:
        pred += lexicon[w] if w in lexicon else 0
    pred = pred / len(tokens)
    return pred


def calculate_statistics(df, tokenizer, lexicon):
    df['review'] = df['review'].apply(lambda x: tokenizer.tokenize(x.lower()))
    df['prediction'] = df['review'].apply(lambda x: calculate_sentiment(x, lexicon))
    threshold = df["prediction"].mean()
    df['prediction'] = df['prediction'].apply(lambda x: -1 if x < threshold else 1)
    print(f'Accuracy: {(df["prediction"] == df["sentiment"]).sum() / len(df["prediction"])}')


def test(lexicon):
    tokenizer = RegexpTokenizer(r'\w+')
    df = parseIMDBDataset()
    print('IMDB results:')
    calculate_statistics(df, tokenizer, lexicon)
    df = parseGameStopDataset()
    print('GameStop results:')
    calculate_statistics(df, tokenizer, lexicon)
