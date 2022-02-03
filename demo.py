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
from sklearn.preprocessing import StandardScaler
import seaborn as sns

def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))

def perform(texts, scores, args):

    start = time.time()
    y = np.array(scores)
    X, vocabulary = generate_bow(texts)
    frequencies = get_frequencies(X)
    print(f'review-word bow matrix generated in {int(time.time() - start)} seconds.')
    start = time.time()
    W = train_linear_pred(X, y)
    print(f'found linear coefficients in {int(time.time() - start)} seconds.')

    glove_words = load_glove_words(EMBEDDINGS_PATH)
    seed_dataset = assign_word_labels(frequencies, W, vocabulary, f_min=args.f_min,
                                      EMBEDDINGS_PATH=EMBEDDINGS_PATH, glove_words=glove_words)
    print('start of training...')
    start = time.time()


    model = train(seed_dataset, run)
    print(f'time of training: {int(time.time() - start)} seconds')
    complete_results = seed_dataset.get_dictionary()

    non_seed_data = {w: 0 for w in glove_words if w not in complete_results}
    non_seed_dataset = SeedDataset(non_seed_data, EMBEDDINGS_PATH, split='test')
    start = time.time()
    results = predict(model, non_seed_dataset)
    print(f'time of predictions: {int(time.time() - start)} seconds')
    complete_results.update(results)
    # close the run on neptune

    complete_results_to_scaled = list(complete_results.values())
    scaled = np.interp(np.array(complete_results_to_scaled),
                       (np.min(complete_results_to_scaled), np.max(complete_results_to_scaled)), (-1, +1)).astype(
        "float32")
    """
    for i, (k, v) in enumerate(complete_results.items()):
        if i > 20:
            break
        print(k, scaled[i])
    """
    indices_high = (-scaled).argsort()[:15]
    indices_low = (scaled).argsort()[:15]
    words = list(complete_results.keys())

    print(f"THE 15 MOST HIGH")
    for i in range(len(indices_high)):
        #if i == 0:
            #run["sys/tags"].add([f"max: {words[indices_high[i]]} : {scaled[indices_high[i]]} "])
        print(f"\n{i} {words[indices_high[i]]} : {scaled[indices_high[i]]}")

    print(f"THE 15 MOST LOW")
    for i in range(len(indices_low)):
        #if i == 0:
            #run["sys/tags"].add([f"min: {words[indices_low[i]]} : {scaled[indices_low[i]]} "])
        print(f"\n{i} {words[indices_low[i]]} : {scaled[indices_low[i]]}")

    mean_value = np.mean(scaled)
    print(f"Mean of the lexicon {mean_value}")
    plt = sns.displot(scaled, kind="kde")
    plt.savefig("Distribution_words_for_score.png")







EMBEDDINGS_PATH = '/content/drive/MyDrive/glove.840B.300d.txt'
if __name__ == '__main__':
    start = time.time()
    file_parameters = open("neptune.json")
    parameters = json.load(file_parameters)
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, required=True, help='Path of the dataset.')
    parser.add_argument('--f_min', type=int, required=True, help='frequency threshold in seed data generation.')
    parser.add_argument('--user', type=str, required=True, help='user to log stuff into his neptune.')
    parser.add_argument('--neg', type=str, required=True, help='more costly method to find better negations.')
    parser.add_argument("--second_extension", type = bool, required=False, help ="use the ablation study version", default=False)
    args = parser.parse_args()

    #start the neptune monitoring
    neptune_parameters = parameters[args.user]

    run = neptune.init(api_token=neptune_parameters["neptune_token"],
                       project=neptune_parameters["neptune_project"])  # pass your credentials

    #in this part we check if we want to perform the second task sa

    if args.second_extension:
        run["sys/tags"].add([f"ablation"])
        years = list(range(1995, 2015))
        clustered_years = list(split(years, 4))
        #for each cluster of years we perform the process
        for cluster in clustered_years:
            texts, scores = parse_dataset(args.dataset_name, True if args.neg == 'complex' else False,
                                          args.second_extension, cluster)
            if len(texts) != 0:
                #TODO manage short dataset
                print(f"CLUSTER {cluster}")
                perform( texts, scores, args)
            else:
                print(f"CLUSTER IS EMPTY")


    else:

        print('the arguments are ', args)
        texts, scores = parse_dataset(args.dataset_name, True if args.neg == 'complex' else False,
                                      args.second_extension)
        print(f'dataset has been read in {int(time.time() - start)} seconds.')

        perform(texts, scores, args)

    run.stop()