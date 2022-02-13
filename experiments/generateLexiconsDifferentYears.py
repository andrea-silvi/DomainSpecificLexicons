from lexiconGeneration import arguments_parsing, createLexicon
import pandas as pd


def generate_list_years_tuples(years_list):
    list_of_tuples_years = []
    for index in range(int(len(years_list) / 2)):
        list_of_tuples_years.append((years_list[2 * index], years_list[2 * index + 1]))
    return list_of_tuples_years


if __name__ == '__main__':
    '''
    Runs experiment 2 over different couples of years. It takes as input a list of the couples of years you want to 
    perform the experiment on e.g. if you wanted to create lexicons for the range of years 1996-2000, 2004-2008, 
    2012-2016 you would input '1996 2000 2004 2008 2012 2016'.
    At the end of the process, it generates a csv named 'scores_per_years.csv' with as rows the vocabulary of the 
    word vectors used, and as columns the scores for each range of years.
    '''
    arguments = arguments_parsing()
    input_string = input("Enter list of years as boundary years separated by space:")
    years = input_string.split()
    if len(years) % 2 != 0:
        raise Exception("Need to input list of years as couple of years.")
    years = [int(year) for year in years]
    couples_of_years = generate_list_years_tuples(years)
    for i, boundary_years in enumerate(couples_of_years):
        lexicon = createLexicon(arguments, list(range(boundary_years[0], boundary_years[1]+1)))
        if i == 0:
            final_dataframe = pd.DataFrame.from_dict(lexicon, orient='index')
            final_dataframe = final_dataframe.sort_index()
        else:
            final_dataframe[i] = (final_dataframe.index).map(lexicon)
    final_dataframe.to_csv(f"scores_per_years.csv")
