from nltk import RegexpTokenizer
from sklearn.metrics import accuracy_score
from dataset.testDatasets import parseIMDBDataset, parseGameStopDataset



def calculate_sentiment(tokens, lexicon):
    '''
    Calculates the sentiment score of the tokens of the input based on the scores in the sentiment lexicon.
    '''
    pred = 0
    for w in tokens:
        pred += lexicon[w] if w in lexicon else 0
    pred = pred / len(tokens)
    return pred


def calculate_statistics(df, tokenizer, lexicon):
    '''
    Calculates sentiment scores of all reviews, calculates the threshold that determines whether a review is positive or
    negative, then calculates the accuracy.
    '''
    df['review'] = df['review'].apply(lambda x: tokenizer.tokenize(x.lower()))
    df['prediction'] = df['review'].apply(lambda x: calculate_sentiment(x, lexicon))
    threshold = df["prediction"].mean()
    df['prediction'] = df['prediction'].apply(lambda x: -1 if x < threshold else 1)
    acc = accuracy_score(df["sentiment"], df["prediction"])
    print(f"Accuracy: {acc}")


def test(lexicon):
    '''
    Runs all the tests of experiment 1 over the IMDB and the GameStop datasets.
    '''
    tokenizer = RegexpTokenizer(r'\w+')
    df = parseIMDBDataset()
    print('IMDB results:')
    calculate_statistics(df, tokenizer, lexicon)
    df = parseGameStopDataset()
    print('GameStop results:')
    calculate_statistics(df, tokenizer, lexicon)
