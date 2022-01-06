import argparse
from AmazonDataset import parse_dataset
from SeedDataset import SeedDataset
from generateSeedData import generate_bow, get_frequencies, train_linear_pred, assign_word_labels
from train import train, predict
import numpy as np
import neptune.new as neptune
import json

from utils.glove_loader import load_glove_words

EMBEDDINGS_PATH = '/content/drive/MyDrive/glove.840B.300d.txt'
if __name__ == '__main__':
    file_parameters = open("neptune.json")
    parameters = json.load(file_parameters)
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, required=True, help='Path of the dataset.')
    parser.add_argument('--f_min', type=int, required=True, help='frequency threshold in seed data generation.')
    parser.add_argument('--user', type=str, required=True, help='user to log stuff into his neptune.')
    args = parser.parse_args()
    print('the arguments are ', args)
    texts, scores = parse_dataset(args.dataset_name)
    print('dataset has been read.')
    y = np.array(scores)
    X, vocabulary = generate_bow(texts)
    frequencies = get_frequencies(X)
    print('review-word bow matrix generated.')
    W = train_linear_pred(X, y)
    print('found linear coefficients.')
    seed_dataset = assign_word_labels(frequencies, W, vocabulary, f_min=args.f_min,
                                      EMBEDDINGS_PATH = EMBEDDINGS_PATH)
    print('start of training...')


    neptune_parameters = parameters[args.user]
    run = neptune.init(api_token=["neptune_token"], project= neptune_parameters["neptune_project"])  # pass your credentials
    model = train(seed_dataset, run)
    complete_results = seed_dataset.get_dictionary()
    glove_words = load_glove_words(EMBEDDINGS_PATH)
    non_seed_data = {w: 0 for w in glove_words if w not in complete_results}
    non_seed_dataset = SeedDataset(non_seed_data, EMBEDDINGS_PATH, split='test')
    results = predict(model, non_seed_dataset)
    complete_results.update(results)
    #close the run on neptune
    run.stop()
    for i, (k, v) in enumerate(complete_results.items()):
        if i > 20:
            break
        print(k, v)

    # save results in a file?
