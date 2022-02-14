# DomainSpecificLexicons 
In this project we expand on the paper 'Domain-Specific Sentiment Lexicons Induced from Labeled Documents' (https://aclanthology.org/2020.coling-main.578/), starting by implementing all of their method from scratch and then performing some additional experiments on top of it. All the experiments and the method description can be found in the report file.

To optimally run this code, one should install:
  - NLTK
  - specifically, download the latest en_core_web_sm (!python -m spacy download en_core_web_sm)
  - SpaCy
  - PyTorch
  - ConvoKit (https://convokit.cornell.edu/documentation/tutorial.html)

## Generate a Lexicon
If you just want to obtain a lexicon from a given input corpus, please run module 'lexiconGeneration.py' from command line with parameters:
  - dataset_name: the path of the input corpus dataset file;
  - f_min: the frequency threshold of the words to keep as seed dataset;
  - neg: the type of negations detection method, with options ['normal', 'whole', 'complex'] (please read the report for more info);
  - weighing: whether to use the negated features linear predictors w_i (again, read the report for more info);
  - exp = 'no_experiment';
  - embeddings = the path of the GloVe word vector file.
For example, in order to generate the lexicon of the Amazon dataset Musical Instruments review dataset, one would launch:

*python3 lexiconGeneration.py --dataset_name='Musical-instruments-dataset-path' --f_min=500 --neg='normal' --exp='no_experiment' --weighing='normal' 
  --embeddings='embeddings_path'*
  
 This will return a python dictionary object, with as keys the words of the vocabulary of the GloVe word vectors and as scores the sentiment polarity scores of each word.

## Run Experiment 1: Unsupervised Review Sentiment Classification
If you want to rerun our experiments on Unsupervised Review Sentiment Classification, one should run the same command as above but with the following modifications:
  - exp = 'exp1'
  - an additional parameter IMDB: the file path of the IMDB dataset to predict the scores of its unseen reviews;
  - an additional parameter GameStop: the file path of the GameStop dataset to predict the scores of its unseen reviews.
For example, one would train on the input corpus Musical Instruments and run the experiment 1 as:

*python3 lexiconGeneration.py --dataset_name='Musical-instruments-dataset-path' --f_min=500 --neg='normal' --exp='exp1' --weighing='normal' 
  --embeddings='embeddings_path' --IMDB='IMDB-dataset-path' --GameStop='GameStop-dataset-path'*
 
 This will build the lexicon and run predictions on the IMDB and GameStop datasets. One could modify the module experiments/test.py in order to predict the sentiment of other unseen datasets fairly easily.
 
## Run Experiment 2: Sentiment Lexicons Evolution over Different Years
If you want to rerun our experiments on Sentiment Lexicons Evolution over Different Years, you should run module 'generateLexiconsDifferentYears.py' from command line with parameters:
  - dataset_name: the path of the input corpus dataset file;
  - f_min: the frequency threshold of the words to keep as seed dataset;
  - neg: the type of negations detection method, with options ['normal', 'whole', 'complex'] (please read the report for more info);
  - weighing: whether to use the negated features linear predictors w_i (again, read the report for more info);
  - exp = 'exp2';
  - embeddings = the path of the GloVe word vector file.
The program will then ask you to insert couples of years in order to separate the input corpus in different groups based on the years the reviews were written in. Please insert the starting and the ending year of all groups by adding spaces in between them. 
For example, if one would want to run experiment 2 on the Movies amazon dataset, by creating 1 lexicon from 1996 to 2000, 1 from 2006 to 2010, and the last one between 2015 and 2018, one would do it like this:

*python3 generateLexiconsDifferentYears.py --dataset_name='Movie-tv-series-dataset-path' --f_min=500 --neg='normal' --exp='exp2' --weighing='normal' 
   --embeddings='embeddings_path' *.
  
Afterwards, when asked one would input the string '*1996 2000 2006 2010 2015 2018*'. Please make sure to insert an even number of years. Also one should be wary that creating a lexicon for a group of years where the reviews are none or a very low number may result in errors/very poor results. 
 
## Run Experiment 3: Reddit Communities Lexicons Comparison
If you finally want to rerun our experiments on Reddit Communities Lexicons Comparison, you should run module 'generateLexiconsSubreddits.py' from command line with parameters:
  - f_min: the frequency threshold of the words to keep as seed dataset;
  - neg: the type of negations detection method, with options ['normal', 'whole', 'complex'] (please read the report for more info);
  - weighing: whether to use the negated features linear predictors w_i (again, read the report for more info);
  - exp = 'exp3';
  - embeddings = the path of the GloVe word vector file.
 The program will then ask you to insert the subreddit names you want to create the lexicons for. Please insert their name (making sure that capitalization is correct) adding spaces in between them.
For example, if one would want to run experiment 3 on the subreddits r/science and r/conspiracy, one would do it like this: 

*python3 generateLexiconsSubreddits.py --f_min=500 --neg='normal' --exp='exp3' --weighing='normal'  --embeddings='embeddings_path'*.

Afterwards, when asked one would input the string '*science conspiracy*'. Please check here https://zissou.infosci.cornell.edu/convokit/datasets/subreddit-corpus/corpus-zipped/ the actual size of the subreddits corpora to see if your machine can run them.

## Authors
You can find our github pages at:
- Ulysse Marquis (Polytechnic of Turin) https://github.com/ulysse1999
- Andrea Silvi (Polytechnic of Turin) https://github.com/andrea-silvi
- Fabio Tatti (Polytechnic of Turin) https://github.com/wrongTactic

  
  

