import pandas as pd


def parseGameStopDataset(dataset_path='/content/drive/MyDrive/gamestop_balanced.csv'):
    df = pd.read_csv(dataset_path)
    df = df.loc[:, ['review_description', 'rating']]
    df = df[(df['rating'] != 3)]
    df['rating'] = df['rating'].apply(lambda x: -1 if x < 3 else 1)
    df.rename(columns={'review_description': 'review', 'rating': 'sentiment'}, inplace=True)
    return df


def parseIMDBDataset(dataset_path='/content/drive/MyDrive/IMDB Dataset.csv'):
    df = pd.read_csv(dataset_path)
    df['sentiment'] = df['sentiment'].apply(lambda x: -1 if x == 'negative' else 1)
    return df
