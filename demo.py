import argparse
from AmazonDataset import parse_dataset
from SeedDataset import SeedDataset
from generateSeedData import generate_bow, get_frequencies, train_linear_pred, assign_word_labels
from train import train, predict
import numpy as np
import neptune.new as neptune
import json
import time
from utils.glove_loader import load_glove_words

EMBEDDINGS_PATH = '/content/drive/MyDrive/glove.840B.300d.txt'
if __name__ == '__main__':
    start = time.time()
    file_parameters = open("neptune.json")
    parameters = json.load(file_parameters)
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, required=True, help='Path of the dataset.')
    parser.add_argument('--f_min', type=int, required=True, help='frequency threshold in seed data generation.')
    parser.add_argument('--user', type=str, required=True, help='user to log stuff into his neptune.')
    args = parser.parse_args()
    print('the arguments are ', args)
    texts, scores = parse_dataset(args.dataset_name)
    print(f'dataset has been read in {time.time()-start}.')
    start = time.time()
    y = np.array(scores)
    X, vocabulary = generate_bow(texts)
    frequencies = get_frequencies(X)
    print(f'review-word bow matrix generated in {time.time()-start}.')
    start = time.time()
    W = train_linear_pred(X, y)
    print(f'found linear coefficients in {time.time()-start}.')

    glove_words = load_glove_words(EMBEDDINGS_PATH)
    seed_dataset = assign_word_labels(frequencies, W, vocabulary, f_min=args.f_min,
                                      EMBEDDINGS_PATH = EMBEDDINGS_PATH, glove_words=glove_words)
    print('start of training...')
    start = time.time()


    neptune_parameters = parameters[args.user]
    run = neptune.init(api_token=neptune_parameters["neptune_token"], project= neptune_parameters["neptune_project"])  # pass your credentials
    model = train(seed_dataset, run)
    print(f'time of training: {time.time()-start}')
    complete_results = seed_dataset.get_dictionary()
    
    non_seed_data = {w: 0 for w in glove_words if w not in complete_results}
    non_seed_dataset = SeedDataset(non_seed_data, EMBEDDINGS_PATH, split='test')
    start = time.time()
    results = predict(model, non_seed_dataset)
    print(f'time of predictions: {time.time()-start}')
    complete_results.update(results)
    #close the run on neptune
    run.stop()
    for i, (k, v) in enumerate(complete_results.items()):
        if i > 20:
            break
        print(k, v)

    # save results in a file?
