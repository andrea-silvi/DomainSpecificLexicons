import argparse
from AmazonDataset import parse_dataset
from generateSeedData import generate_bow, train_linear_pred, assign_word_labels
from train import train, predict
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, required=True, help='Path of the dataset.')
    parser.add_argument('--f_min', type=int, required=True, help='frequency threshold in seed data generation.')
    args = parser.parse_args()
    print('the arguments are ', args)
    texts, scores = parse_dataset(args.dataset_name)
    print('dataset has been read.')
    y = np.array(scores)
    X, vocabulary = generate_bow(texts)
    print('review-word bow matrix generated.')
    W = train_linear_pred(X, y)
    print('found linear coefficients.')
    seed_dataset, non_seed_dataset = assign_word_labels(X, W, vocabulary, f_min=args.f_min)
    print('start of training...')
    model = train(seed_dataset)
    complete_results = seed_dataset.get_dictionary()
    results = predict(model, non_seed_dataset)
    complete_results.update(results)
    for i, (k, v) in enumerate(complete_results.items()):
        if i > 20:
            break
        print(k, v)

    # save results in a file?
