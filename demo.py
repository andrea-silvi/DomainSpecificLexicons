from AmazonDataset import parseDataset
from generateSeedData import generate_bow, train_linear_pred
import numpy as np


reviews = parseDataset("/content/drive/MyDrive/dataset/Musical_Instruments_5.json.gz")

X, y, vocabulary = generate_bow(reviews)

W = train_linear_pred(X, y)

