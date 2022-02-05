from nltk import RegexpTokenizer
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
from testDatasets import parseIMDBDataset, parseGameStopDataset
import numpy as np

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
    fraction_negatives = (df["sentiment"] == -1).sum()/len(df["sentiment"])
    threshold = np.percentile(df['prediction'], 100*fraction_negatives)
    df['prediction'] = df['prediction'].apply(lambda x: -1 if x < threshold else 1)

    acc = accuracy_score(df["sentiment"], df["prediction"])
    pr = precision_score(df["sentiment"], df["prediction"], average='macro')
    rec = recall_score(df["sentiment"], df["prediction"], average='macro')
    fscore = f1_score(df["sentiment"], df["prediction"], average='macro')

    print(f"Accuracy : {acc}")
    print(f"Precision : {pr}")
    print(f"Recall : {rec}")
    print(f"F1 score : {fscore}")


def test(lexicon):
    tokenizer = RegexpTokenizer(r'\w+')
    df = parseIMDBDataset()
    print('IMDB results:')
    calculate_statistics(df, tokenizer, lexicon)
    df = parseGameStopDataset()
    print('GameStop results:')
    calculate_statistics(df, tokenizer, lexicon)


