import argparse
from AmazonDataset import parse_dataset
from SeedDataset import SeedDataset
from generateSeedData import generate_bow, get_frequencies, train_linear_pred, assign_word_labels
from train import train, predict
from test import test
import numpy as np
import neptune.new as neptune
import json
import time
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
                        choices=['normal', 'whole'], default='normal')
    parser.add_argument('--weighing', type=str, required=False, 
                        help='different methods to compute words scores knowing its negation score',
                        default='normal', choices=['normal', "whole", "complex"])
    parser.add_argument('--exp', type=str, required=True, help='Type of experiment.',
                        choices=['exp1', 'exp2', 'exp3'])
    args = parser.parse_args()
    print('the arguments are ', args)
    return args



EMBEDDINGS_PATH = '/content/drive/MyDrive/glove.840B.300d.txt'


if __name__ == '__main__':
    
    with open("neptune.json") as neptune_file:
        parameters = json.load(neptune_file)

    args = cli_parsing()

    texts, scores = parse_dataset(args.dataset_name, args.neg)
    
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
    
    results = predict(model, non_seed_dataset)
    
    complete_results.update(results)
    if args.exp == 'exp1':
        test(lexicon=complete_results)
    """
    complete_results_to_scaled = list(complete_results.values())
    scaled = np.array(complete_results_to_scaled)
    for i, (k, v) in enumerate(complete_results.items()):
        if i > 20:
            break
        print(k, scaled[i])
    
    indices_high = (-scaled).argsort()[:15]
    indices_low = (scaled).argsort()[:15]
    words = list(complete_results.keys())

    print(f"THE 15 MOST HIGH")
    for i in range(len(indices_high)):
        if i == 0:
            run["sys/tags"].add([f"max: {words[indices_high[i]]} : {scaled[indices_high[i]]} "])
        print(f"\n{i} {words[indices_high[i]]} : {scaled[indices_high[i]]}")

    print(f"THE 15 MOST LOW")
    for i in range(len(indices_low)):
        if i == 0:
            run["sys/tags"].add([f"min: {words[indices_low[i]]} : {scaled[indices_low[i]]} "])
        print(f"\n{i} {words[indices_low[i]]} : {scaled[indices_low[i]]}")

    mean_value = np.mean(scaled)
    print(f"Mean of the lexicon {mean_value}")
    plt = sns.displot(scaled, kind="kde")
    plt.savefig("Distribution_words_for_score.png")
    
    run["sys/tags"].add([f"f-min: {args.f_min}", f"mean: {mean_value}"])
    """
    run.stop()
