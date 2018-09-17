import tensorflow as tf
from tensorflow.contrib.keras.api.keras.datasets.mnist import load_data as load_mnist_data
import numpy as np
import numpy.random as nprd
import pickle

class DataLoader(object):
    def __init__(self, dataset, batch_size, file_path="./dataset/"):
        dataset_handler = {}
        dataset_handler["cifar10"] = self.cifar10_handler
        dataset_handler["mnist"] = self.mnist_handler

        self.shape = [32, 32, 3]
        self.batch_size = batch_size
        self.file_path = file_path
        self.data_pool = []
        self.batch_idx = 0
        self.batch_count = 0

        if dataset not in dataset_handler:
            print("invalid dataset parameter %s" % dataset)
            raise -1
        else:
            dataset_handler[dataset]()

    def cifar10_handler(self):
        def unpickle(file):
            with open(file, 'rb') as fo:
                dict = pickle.load(fo, encoding='bytes')
            return dict
        self.shape = [32, 32, 3]
        self.file_path += 'cifar10/'
        self.batch_count = int(60000 / self.batch_size)
        for file_name in (["data_batch_%d" % i for i in range(1, 6)] + ["test_batch"]):
            self.data_pool += [unpickle(self.file_path + file_name)[b'data']]
        self.data_pool = np.concatenate(self.data_pool, axis=0)
        self.data_pool = np.reshape(self.data_pool, [60000, 3, 32, 32])
        self.data_pool = np.transpose(self.data_pool, axes=[0, 2, 3, 1])
        np.random.shuffle(self.data_pool)
        # Now data_pool -> [60000, 32, 32, 3]
    def mnist_handler(self):
        (p0, _), (p1, _) = load_mnist_data()
        self.shape = [28, 28, 1]
        self.data_pool = np.concatenate([p0, p1], axis=0)
        self.data_pool = np.expand_dims(self.data_pool, axis=-1)
        self.batch_count = int(70000 / self.batch_size)
        np.random.shuffle(self.data_pool)

    def next_batch(self):
        ret_v = self.data_pool[self.batch_idx * self.batch_size:(self.batch_idx + 1) * self.batch_size]
        self.batch_idx += 1
        if self.batch_idx >= self.batch_count:
            self.reset_pointer()
        return ret_v

    def reset_pointer(self):
        self.batch_idx = 0
        nprd.shuffle(self.data_pool)