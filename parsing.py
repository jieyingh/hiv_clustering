import argparse
import os

def file_path(path):
    if os.path.isfile(path):
        return path
    else:
        raise FileNotFoundError(path)

def parse():

    parser = argparse.ArgumentParser()
    parser.add_argument('x', type=file_path,
                        help='Sequences to be classified')
    parser.add_argument('y', type=file_path,
                        help='Class of sequences')
    parser.add_argument('-k', '--kmer', default=6, type=int,
                        help='Number of k-mers to generate')
    parser.add_argument('-m', '--method', default='all', choices=['knn', 'dt', 'svc', 'mlp', 'all'],
                        help='Method to use for classification')
    parser.add_argument('-s', '--show', action='store_true', default=True,
                        help='Show plots')   
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='adds verbosity')

    args = parser.parse_args()

    return args

# args = parse()
# print(args)
