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

*!python3 lexiconGeneration.py --dataset_name='Musical-instruments-dataset-path' --f_min=500 --neg='normal' --exp='no_experiment' --weighing='normal' 
  --embeddings='embeddings_path'*.
  
  This will return a python dictionary object, with as keys the words of the vocabulary of the GloVe word vectors and as scores the sentiment polarity scores of each word.

## Run Experiment 1: Unsupervised Review Sentiment Classification
If you want to rerun our experiments on Unsupervised Review Sentiment Classification, one should run the same command as above but with the following modifications:
  - exp = 'exp1'
  - an additional parameter IMDB: the file path of the IMDB dataset to predict the scores of its unseen reviews;
  - an additional parameter GameStop: the file path of the GameStop dataset to predict the scores of its unseen reviews.
For example, one would train on the input corpus Musical Instruments and run the experiment 1 as:

*!python3 lexiconGeneration.py --dataset_name='Musical-instruments-dataset-path' --f_min=500 --neg='normal' --exp='exp1' --weighing='normal' 
  --embeddings='embeddings_path' --IMDB='IMDB-dataset-path' --GameStop='GameStop-dataset-path'*
  

