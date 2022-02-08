from demo import cli_parsing, createLexicon
import time
import pandas as pd



if __name__ == '__main__':
    start = time.time()
    arguments = cli_parsing()
    if arguments.subreddits is not None:
        subreddits = arguments.subreddits
    else:
        print("ATTENTION! List of subreddits to build lexicons of needed. Restart.")
    # for each subreddit we create the lexicon
    for i, subreddit in enumerate(subreddits):
        lexicon = createLexicon(arguments, subreddit)
        if i == 0:
            final_dataframe = pd.DataFrame.from_dict(lexicon, orient='index', columns=[subreddit])
        else:
            for word, sentiment in lexicon.items():
                final_dataframe.loc[word, subreddit] = sentiment
    final_dataframe.to_csv(f"scores_per_subreddit.csv")