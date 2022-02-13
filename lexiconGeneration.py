import argparse
from dataset.AmazonDataset import parse_dataset, parse_dataset_by_year
from dataset.SeedDataset import SeedDataset
from seedDataInduction.seedDataInduction import generate_bow, get_frequencies, train_linear_pred, assign_word_labels
from dataset.subredditDataset import parse_subreddit
from neuralLabelExpansion.train import train, predict
from experiments.test import test
import numpy as np
from utils_.glove_loader import load_glove_words


def arguments_parsing():
    '''
    parses and manages command line arguments.
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, required=True, help='Path of the dataset.')
    parser.add_argument('--f_min', type=int, required=True, help='frequency threshold in seed data generation.',
                        default=500)
    parser.add_argument('--neg', type=str, required=True, help='different methods to find negations.',
                        choices=['normal', 'whole', 'complex'], default='normal')
    parser.add_argument('--weighing', type=str, required=False,
                        help='either not use or use scores of negated word in seed data induction step.',
                        default='normal', choices=['normal', "whole"])
    parser.add_argument('--exp', type=str, required=True, help='Type of experiment.',
                        choices=['exp1', 'exp2', 'exp3'])
    parser.add_argument('--embeddings', type=str, required=True, help='file path of word vector embeddings.')
    args = parser.parse_args()
    return args


def createLexicon(args, years=None, subreddit=None):
    print('Starting lexicon generation with input corpus parsing...')
    if args.exp == 'exp1':
        texts, scores = parse_dataset(args.dataset_name, args.neg)
    elif args.exp == 'exp2':
        texts, scores = parse_dataset_by_year(args.dataset_name, years, args.neg)
    else:
        texts, scores = parse_subreddit(subreddit)
    y = np.array(scores)
    print('Generating Bag of words model of input corpus...')
    X, vocabulary = generate_bow(texts)
    frequencies = get_frequencies(X)
    print('Training linear predictor...')
    W = train_linear_pred(X, y)
    glove_words = load_glove_words(args.embeddings)
    print('Creating seed dataset...')
    seed_dataset = assign_word_labels(frequencies, W, vocabulary,
                                      f_min=args.f_min,
                                      embeddings_path=args.embeddings,
                                      glove_words=glove_words,
                                      weighing=args.weighing)
    print(f'Start of training with {len(seed_dataset)} seed words...')
    model = train(seed_dataset)
    complete_results = seed_dataset.get_dictionary()
    non_seed_data = {w: 0 for w in glove_words if w not in complete_results}
    non_seed_dataset = SeedDataset(non_seed_data, args.embeddings, split='test')
    print(f'Starting label expansion phase...')
    results = predict(model, non_seed_dataset)
    complete_results.update(results)
    if args.exp == 'exp1':
        test(lexicon=complete_results)
    else:
        return complete_results


if __name__ == '__main__':
    arguments = arguments_parsing()
    createLexicon(args=arguments)
