import argparse
from functools import wraps
import time


def timing_wrapper(message):
    def f(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            res = func(*args, **kwargs)
            print(message, f': {int(time.time() - start)} seconds.')
            return res

        return wrapper

    return f


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
    parser.add_argument('--IMDB', type=str, required=False, help='file path of IMDB dataset for experiment 1.')
    parser.add_argument('--GameStop', type=str, required=False, help='file path of GameStop dataset for experiment 1.')
    args = parser.parse_args()
    return args