import numpy as np
import scipy.io as matlibio

from experiments import init_experiment, QuickParams, DefaultParams, FineParams
from observations import DataMatrix


###
### First follow the configuration directions in README.md. Then run the following:
###
###     python example.py
###     python experiments.py everything example
###

def read_array_matlib(fname):
    x = matlibio.loadmat(fname)
    return np.transpose(x['IMAGES'], axes=[2, 0, 1])

def collect_patches(image):
    collection = np.zeros([1000, 144])
    for i in range(1000):
        index = int(np.random.uniform() * 10.0)
        patch_x = int(np.random.uniform() * 500.0)
        patch_y = int(np.random.uniform() * 500.0)
        collection[i] = np.reshape(image[index, patch_x:patch_x+12, patch_y:patch_y+12], [144])
    return collection

def read_list(fname):
    return list(map(str.strip, open(fname).readlines()))


def init():
    X = read_array_matlib('example_data/image_patch/image_patch.mat')
    X = collect_patches(X)
    print(X.shape)
    row_labels = None  # read_list('example_data/animals-names.txt')
    col_labels = None  # read_list('example_data/animals-features.txt')

    # normalize to zero mean, unit variance

    X -= X.mean()
    X /= X.std()
    # since the data were binary, add a small amount of noise to prevent degeneracy
    X = np.random.normal(X, np.sqrt(0.1))

    data_matrix = DataMatrix.from_real_values(X)  # , row_labels=row_labels, col_labels=col_labels)
    init_experiment('image_patch', data_matrix, QuickParams(search_depth=3))


if __name__ == '__main__':
    init()



