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
    print(f'first row: {df.loc[0, "review"]}, {df.loc[0, "prediction"]}, {df.loc[0, "sentiment"]}')
    fraction_negatives = (df["sentiment"] == -1).sum()/len(df["sentiment"])
    threshold = np.percentile(df['prediction'], 100*fraction_negatives)
    df['prediction'] = df['prediction'].apply(lambda x: -1 if x < threshold else 1)
    print(f'first row new prediction: {df.loc[0, "prediction"]}')
    print(f'Accuracy: {(df["prediction"] == df["sentiment"]).sum() / len(df["prediction"])}')
    print(f'n positives vs actual positives: {(df["prediction"]==1).sum()} vs {(df["sentiment"]==1).sum()}')
    print(f'n negatives vs actual negatives: {(df["prediction"]==-1).sum()} vs {(df["sentiment"]==-1).sum()}')
    scores = precision_recall_fscore_support(df["sentiment"], df["prediction"])
    print(f'Negatives Precision: {scores[0][0]}, Recall: {scores[1][0]}, F-Score: {scores[2][0]}')
    print(f'Positives Precision: {scores[0][1]}, Recall: {scores[1][1]}, F-Score: {scores[2][1]}')

def test(lexicon):
    tokenizer = RegexpTokenizer(r'\w+')
    df = parseIMDBDataset()
    print('IMDB results:')
    calculate_statistics(df, tokenizer, lexicon)
    df = parseGameStopDataset()
    print('GameStop results:')
    calculate_statistics(df, tokenizer, lexicon)


