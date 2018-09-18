import tensorflow as tf
import tensorflow.contrib as tc
import tensorflow.contrib.layers as tcl
import PIL.Image as image
from data_loader import DataLoader
from tensorflow.contrib.layers import xavier_initializer
leaky_relu = tf.nn.leaky_relu
import numpy as np
xavier = xavier_initializer()

from experiments import init_experiment, QuickParams, LargeParams
from observations import DataMatrix

###
### First follow the configuration directions in README.md. Then run the following:
###
###     python example.py
###     python experiments.py everything example
###

def read_array(fname):
    return np.array([map(float, line.split()) for line in open(fname)])

def read_list(fname):
    return map(str.strip, open(fname).readlines())

def init():
    X = read_array('example_data/animals-data.txt')
    row_labels = read_list('example_data/animals-names.txt')
    col_labels = read_list('example_data/animals-features.txt')

    # normalize to zero mean, unit variance
    X -= X.mean()
    X /= X.std()

    # since the data were binary, add a small amount of noise to prevent degeneracy
    X = np.random.normal(X, np.sqrt(0.1))

    data_matrix = DataMatrix.from_real_values(X, row_labels=row_labels, col_labels=col_labels)
    init_experiment('example', data_matrix, QuickParams(search_depth=2))

class Encoder(object):
    def __init__(self, name='encoder', dim=32):
        self.x_dim = 784
        self.dim = dim
        self.name = name

    def __call__(self, x, reuse=tf.AUTO_REUSE):
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE) as vs:
            bs = tf.shape(x)[0]
            x = tf.reshape(x, [bs, 28, 28, 1])
            conv1 = tc.layers.convolution2d(
                x, 16, [3, 3], [1, 1],
                weights_initializer=xavier,
                activation_fn=tf.identity
            )
            conv1 = leaky_relu(conv1, alpha=0.2)
            conv2 = tc.layers.convolution2d(
                conv1, 32, [4, 4], [2, 2],
                weights_initializer=xavier,
                activation_fn=tf.identity
            )
            conv2 = leaky_relu(conv2, alpha=0.2)
            conv3 = tc.layers.convolution2d(
                conv2, 64, [3, 3], [1, 1],
                weights_initializer=xavier,
                activation_fn=tf.identity
            )
            conv3 = leaky_relu(conv3, alpha=0.2)
            conv4 = tc.layers.convolution2d(
                conv3, 64, [4, 4], [2, 2],
                weights_initializer=xavier,
                activation_fn=tf.identity
            )
            conv4 = leaky_relu(conv4, alpha=0.2)
            conv5 = tc.layers.convolution2d(
                conv4, 128, [4, 4], [2, 2],
                weights_initializer=xavier,
                activation_fn=tf.identity
            )
            conv5 = leaky_relu(conv5, alpha=0.2)
            conv6 = tc.layers.convolution2d(
                conv5, 256, [3, 3], [1, 1],
                weights_initializer=xavier,
                activation_fn=tf.identity
            )
            conv6 = leaky_relu(conv6, alpha=0.2)
            fc1 = tcl.flatten(conv6)
            fc1 = tc.layers.fully_connected(fc1, self.dim, activation_fn=tf.identity)
            return fc1

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]

def save_img(batch_of_img, batch_size, channel=1, file="example"):
    imgs = np.transpose(batch_of_img, [0, 3, 1, 2])
    for i in range(batch_size):
        img = imgs[i]
        img = np.asarray(img, dtype=np.uint8)
        if channel == 3:
            r = image.fromarray(img[0]).convert('L')
            g = image.fromarray(img[1]).convert('L')
            b = image.fromarray(img[2]).convert('L')
        else:
            r = image.fromarray(img[0]).convert('L')
            g = r
            b = r
        img = image.merge("RGB", (r, g, b))
        img.save('./%s/sample%d.png' % (file, i), 'png')

class Model(object):
    def __init__(self):
        # Placeholders:
        self.real = tf.placeholder(shape=[None, 28, 28, 1], dtype=tf.float32)
        real = self.real / 128.0 - 1.0
        encoder = Encoder()
        self.code = encoder(real)

model_path = "./saved_model/MLP_G"
BATCH_SIZE = 100
LOAD = True
def main():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    data_loader = DataLoader("mnist", BATCH_SIZE)
    vae = Model()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(tf.global_variables())
    if LOAD:
        saver.restore(sess, model_path)
    USED_NUM = 2000
    code_mat = np.zeros([USED_NUM// BATCH_SIZE, BATCH_SIZE, 32])
    imgs = []
    for iter_idx in range(USED_NUM // BATCH_SIZE):
        ref = data_loader.next_batch()
        imgs.append(ref)
        code = sess.run(vae.code, feed_dict={vae.real: ref})
        if iter_idx % 10 == 0:
            print(code)
        code_mat[iter_idx] = code
    imgs = np.concatenate(imgs, axis=0)
    save_img(imgs, USED_NUM, file='data/mnist_img')
    code_mat = np.reshape(code_mat, [USED_NUM, 32])
    code_mat /= np.std(code_mat)
    code_mat -= np.mean(code_mat)
    # code_mat = np.random.normal(code_mat, np.sqrt(0.1))
    data_matrix = DataMatrix.from_real_values(code_mat)
    init_experiment('mnist-noised', data_matrix, QuickParams(search_depth=2))

if __name__ == "__main__":
    main()
