import argparse
from AmazonDataset import parse_dataset
from generateSeedData import generate_bow, train_linear_pred, assign_word_labels
from train import train, predict
import numpy as np

# reviews, scores = parse_dataset("/content/drive/MyDrive/dataset/Musical_Instruments_5.json.gz")

# y = np.array(scores)

# X, vocabulary = generate_bow(reviews)

# W = train_linear_pred(X, y)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, required=True, help='Path of the dataset.')
    parser.add_argument('--f_min', type=int, required=True, help='frequency threshold in seed data generation.')
    args = parser.parse_args()
    texts, scores = parse_dataset(args.dataset_name)
    y = np.array(scores)
    X, vocabulary = generate_bow(texts)
    W = train_linear_pred(X, y)
    seed_dataset, non_seed_dataset = assign_word_labels(X, W, vocabulary, f_min=args.f_min)
    model = train(seed_dataset)
    original_results = seed_dataset.get_dictionary()
    results = predict(model, non_seed_dataset)
    complete_results = original_results | results #needs Python 3.9!

    # save results in a file?
