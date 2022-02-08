from demo import cli_parsing, createLexicon
import time
import pandas as pd

# def split(a, n):
# k, m = divmod(len(a), n)
# return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))

"""
def perform(texts, scores, args, cluster = None):

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
    max_n = min(len(scaled), 10)

    indices_high = (-scaled).argsort()[:max_n]
    indices_low = (scaled).argsort()[:max_n]
    words = list(complete_results.keys())

    #save the top and bottom words score pairs
    top = []
    bottom = []
    print(f"THE 15 MOST HIGH")
    for i in range(len(indices_high)):
        #if i == 0:
            #run["sys/tags"].add([f"max: {words[indices_high[i]]} : {scaled[indices_high[i]]} "])
        print(f"\n{i} {words[indices_high[i]]} : {scaled[indices_high[i]]}")
        top.append([words[indices_high[i]],scaled[indices_high[i]] ])

    print(f"THE 15 MOST LOW")
    for i in range(len(indices_low)):
        #if i == 0:
            #run["sys/tags"].add([f"min: {words[indices_low[i]]} : {scaled[indices_low[i]]} "])
        print(f"\n{i} {words[indices_low[i]]} : {scaled[indices_low[i]]}")
        bottom.append([words[indices_low[i]],scaled[indices_low[i]] ])

    mean_value = np.mecan(scaled)
    print(f"Mean of the lexicon {mean_value} ")

    #we filter out the values [-0. 2, 0.2]
    distribution_filtered = list(filter(lambda x: (x > mean_value+0.2) and (x < mean_value - 0.2), scaled))
    plt = sns.displot(distribution_filtered, kind="kde")
    if cluster == None:
        plt.savefig("Distribution_words_for_score.png")
    else:
        plt.savefig(f"Distribution_words_for_score_{cluster}.png")
    return cluster, top, bottom

"""

if __name__ == '__main__':
    start = time.time()
    arguments = cli_parsing()
    # years = list(range(1995, 2015))
    # clustered_years = list(split(years, 4))

    years = [1996, 2002, 2002, 2008, 2008, 2014, 2015, 2018]
    couples_of_years = []
    for i in range(0, len(years), 2):
        couples_of_years.append((years[i], years[i+1]))

    # for each cluster of years we perform the process
    for i, boundary_years in enumerate(couples_of_years):
        lexicon = createLexicon(arguments, list(range(boundary_years[0], boundary_years[1]+1)))
        if i == 0:
            final_dataframe = pd.DataFrame.from_dict(lexicon, orient='index')
        else:
            for word, sentiment in lexicon.items():
                final_dataframe.loc[word, i] = sentiment
    final_dataframe.to_csv(f"scores_per_year_pairs.csv")
