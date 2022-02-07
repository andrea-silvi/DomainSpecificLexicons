import argparse
from AmazonDataset import parse_dataset, parse_dataset_by_year
from SeedDataset import SeedDataset
from generateSeedData import generate_bow, get_frequencies, train_linear_pred, assign_word_labels
from train import train, predict
from test import test
import numpy as np
import neptune.new as neptune
import json
import time
import gc
from utils.glove_loader import load_glove_words
from utils.utils import timing_wrapper
from sklearn.preprocessing import StandardScaler
import seaborn as sns


def cli_parsing():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, required=True, help='Path of the dataset.')
    parser.add_argument('--f_min', type=int, required=True, help='frequency threshold in seed data generation.',
                        default=500)
    parser.add_argument('--user', type=str, required=True, help='user to log stuff into his neptune.',
                        choices=['Ulysse', 'Andrea', 'Fabio'])
    parser.add_argument('--neg', type=str, required=True, help='different methods to find negations.',
                        choices=['normal', 'whole', 'complex'], default='normal')
    parser.add_argument('--weighing', type=str, required=False, 
                        help='different methods to compute words scores knowing its negation score',
                        default='normal', choices=['normal', "whole"])
    parser.add_argument('--exp', type=str, required=True, help='Type of experiment.',
                        choices=['exp1', 'exp2', 'exp3'])
    args = parser.parse_args()
    print('the arguments are ', args)
    return args



EMBEDDINGS_PATH = '/content/drive/MyDrive/glove.840B.300d.txt'


def createLexicon(args, cluster=None):
    with open("neptune.json") as neptune_file:
        parameters = json.load(neptune_file)
    if cluster is None:
        texts, scores = parse_dataset(args.dataset_name, args.neg)
    else:
        texts, scores = parse_dataset_by_year(args.dataset_name, cluster, args.neg)
    y = np.array(scores)
    X, vocabulary = generate_bow(texts)
    frequencies = get_frequencies(X)
    W = train_linear_pred(X, y)
    glove_words = load_glove_words(EMBEDDINGS_PATH)
    seed_dataset = assign_word_labels(frequencies, W, vocabulary,
                                      f_min=args.f_min,
                                      EMBEDDINGS_PATH=EMBEDDINGS_PATH,
                                      glove_words=glove_words,
                                      weighing=args.weighing)
    print('start of training...')
    neptune_parameters = parameters[args.user]
    run = neptune.init(api_token=neptune_parameters["neptune_token"],
                       project=neptune_parameters["neptune_project"])  # pass your credentials
    model = train(seed_dataset, run)
    complete_results = seed_dataset.get_dictionary()
    non_seed_data = {w: 0 for w in glove_words if w not in complete_results}
    non_seed_dataset = SeedDataset(non_seed_data, EMBEDDINGS_PATH, split='test')

    # attempt at saving memory
    del(seed_dataset)
    gc.collect()

    results = predict(model, non_seed_dataset)
    complete_results.update(results)
    run.stop()

    if args.exp == 'exp1':
        test(lexicon=complete_results)
    elif args.exp == 'exp2':
        return complete_results

if __name__ == '__main__':
    arguments = cli_parsing()
    createLexicon(args=arguments)
