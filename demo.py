import argparse
from AmazonDataset import parse_dataset
from generateSeedData import generate_bow, get_frequencies, train_linear_pred, assign_word_labels
from train import train, predict
import numpy as np
import neptune.new as neptune
import json

if __name__ == '__main__':
    file_parameters = open("parameters.json")
    parameters = json.load(file_parameters)
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, required=True, help='Path of the dataset.')
    parser.add_argument('--f_min', type=int, required=True, help='frequency threshold in seed data generation.')
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
    seed_dataset, non_seed_dataset = assign_word_labels(frequencies, W, vocabulary, f_min=args.f_min)
    print('start of training...')

    #neptune definition of the run
    neptune_token = parameters["neptune_token"]
    neptune_user = parameters["neptune_user"]

    if neptune_user == "fabio":
        neptune_project = "fbtattix/DomainSpecificLexicon"

    elif neptune_user == "ulysse":
        pass
        #TODO: put your neptune project in order to make it work on your neptune (to select neptune user use parameters.json)

    elif neptune_user == "andrea":
        pass
        #TODO: put your neptune project in order to make it work on your neptune (to select neptune user use parameters.json)

    run = neptune.init(api_token=neptune_token, project= neptune_project)  # pass your credentials

    model = train(seed_dataset, run)
    complete_results = seed_dataset.get_dictionary()
    results = predict(model, non_seed_dataset)
    complete_results.update(results)
    #close the run on neptune
    run.stop()
    for i, (k, v) in enumerate(complete_results.items()):
        if i > 20:
            break
        print(k, v)

    # save results in a file?
