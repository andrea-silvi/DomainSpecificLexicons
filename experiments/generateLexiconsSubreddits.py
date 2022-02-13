from lexiconGeneration import arguments_parsing, createLexicon
import pandas as pd

if __name__ == '__main__':
    '''
        Runs experiment 3 over different subreddits. It takes as input a list of the subreddits you want to 
        perform the experiment on e.g. if you wanted to create lexicons for the subreddits r/science and r/conspiracy
         you would input 'science conspiracy'.
        At the end of the process, it generates a csv named 'scores_per_subreddit.csv' with as rows the vocabulary of 
        the word vectors used, and as columns the scores for each subreddit.
        Check 'https://zissou.infosci.cornell.edu/convokit/datasets/subreddit-corpus/corpus-zipped/' to find all the 
        subreddits available in the ConvoKit package.
        '''
    arguments = arguments_parsing()
    input_string = input("Enter list of subreddits separated by space:")
    subreddits = input_string.split()
    for i, subreddit in enumerate(subreddits):
        lexicon = createLexicon(arguments, subreddit=subreddit)
        if i == 0:
            final_dataframe = pd.DataFrame.from_dict(lexicon, orient='index', columns=[subreddit])
        else:
            for word, sentiment in lexicon.items():
                final_dataframe.loc[word, subreddit] = sentiment
    final_dataframe.to_csv(f"scores_per_subreddit.csv")