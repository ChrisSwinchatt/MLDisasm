import sys

try:
    import ujson as json
except ModuleNotFoundError:
    import json

import matplotlib.pyplot as plt

import numpy as np

def compute_correlation(file):
    '''
    Compute the Pearson correlation coefficient for a training set.
    :param file: An open file containing the training set.
    :returns: A tuple of the input lengths, target lengths and correlation.
    '''
    X_lens   = []
    y_lens   = []
    line_num = 1
    for line in file:
        X, y = json.loads(line)
        X = list(filter(lambda xi: xi[0] >= 0, X))
        y = list(filter(lambda yi: yi[0] >= 0, y))
        X_lens.append(len(X))
        y_lens.append(len(y))
        if line_num % 25000 == 0:
            sys.stdout.write('.')
            sys.stdout.flush()
        line_num += 1
    sys.stdout.write('\n')
    pearson = np.corrcoef(X_lens, y_lens)[0, 1]
    return X_lens, y_lens, pearson

if __name__ == '__main__:
    path = 'data/att/training.json'
    with open(path) as file:
        print('Loading...')
        X, y, coeff = compute_correlation(file)
        print('coeff =', coeff)

    print('Plotting...')
    plt.clf()
    plt.xlabel('Length of input vector')
    plt.ylabel('Length of output vector')
    plt.xticks(range(min(X), max(X) + 1))
    plt.yticks(range(min(y), max(y) + 1, 2))
    plt.scatter(X, y)
    print('Saving...')
    plt.savefig('corr.png')
