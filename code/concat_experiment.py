import numpy as np

from experiments import init_experiment, QuickParams, DefaultParams, FineParams
from observations import DataMatrix


###
### First follow the configuration directions in README.md. Then run the following:
###
###     python example.py
###     python experiments.py everything example
###

def read_array(fname):
    return np.array([list(map(float, line.split())) for line in open(fname)])


def read_list(fname):
    return list(map(str.strip, open(fname).readlines()))


def init():
    nrows = 500
    ncomp = 10
    pi_crp = np.ones(ncomp) / ncomp
    U = np.random.multinomial(1, pi_crp, size=nrows)
    V = np.random.normal(size=[nrows, 6]) * np.array([[3.0, 1.41, 5.9, 2.6, 8.97, 9.32]])
    print(U.shape)
    print(V.shape)
    X = np.concatenate((U, V), axis=-1)
    row_labels = None  # read_list('example_data/animals-names.txt')
    col_labels = None  # read_list('example_data/animals-features.txt')

    # normalize to zero mean, unit variance

    X -= X.mean()
    X /= X.std()
    # since the data were binary, add a small amount of noise to prevent degeneracy
    X = np.random.normal(X, np.sqrt(0.1))

    data_matrix = DataMatrix.from_real_values(X)  # , row_labels=row_labels, col_labels=col_labels)
    init_experiment('concat', data_matrix, FineParams(search_depth=3))


if __name__ == '__main__':
    init()



