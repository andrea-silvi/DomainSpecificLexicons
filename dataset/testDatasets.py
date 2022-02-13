import pandas as pd


def parseGameStopDataset(dataset_path):
    df = pd.read_csv(dataset_path)
    return df


def parseIMDBDataset(dataset_path):
    df = pd.read_csv(dataset_path)
    df['sentiment'] = df['sentiment'].apply(lambda x: -1 if x == 'negative' else 1)
    return df
