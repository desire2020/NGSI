import tensorflow as tf
import tensorflow.contrib.layers as tcl
import numpy as np
nax = np.newaxis
import tensorflow.contrib.autograph as autograph
import tensorflow.contrib.distributions as tfdist
import time
import PIL.Image as image
import gc
from config import *
from CNNGuider import Classifier
import os

def save_img(batch_of_img, batch_size, channel=1, file="data/results/confusion.png"):
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
        img.save('./%s' % (file), 'png')

dictionary = {
    "G": 0,
    "M": 1,
    "B": 2,
    "C": 3,
    "'": 4,
    "e": 5,
    "@": 6,
    "+": 7,
    "*": 8
}
induction_rule = {
    "G": ["+@G'GG", "*eGG", "+@MGG", "+@G'MG",
          "+@BGG", "+@G'BG", "+@CGG", "+@G'CG"],
    "M": ["+@MGG", "B"],
    "B": ["+@BGG"]
}

# Synthetic Data Table
def np_onehot(index, depth):
    ret = np.zeros([depth])
    ret[index] = 1.0
    return ret
def check_and_handle_uninit(sess):
    vl = tf.global_variables()
    p = [tf.is_variable_initialized(v) for v in vl]
    p = sess.run(p)
    v = [vl[i] for i in range(len(vl)) if not p[i]]
    sess.run(tf.variables_initializer(v))

log_file = open("saved_model/log.txt", "w")

def generate_ar(nrows, ncols, a):
    X = np.zeros((nrows, ncols))
    X[0,:] = np.random.normal(size=ncols)
    for i in range(1, nrows):
        X[i,:] = a * X[i-1,:] + np.random.normal(0., np.sqrt(1-a**2), size=ncols)
    return X

def generate_data(data_str, nrows, ncols, ncomp, return_components=False):
    IBP_ALPHA = 2.
    pi_crp = np.ones(ncomp) / ncomp
    pi_ibp = np.ones(ncomp) * IBP_ALPHA / ncomp
    data, components = None, None

    if data_str[-1] == 'T':
        data_str = data_str[:-1]
        transpose = True
        nrows, ncols = ncols, nrows
    else:
        transpose = False

    if data_str == 'low-rank':
        U = np.random.normal(0., 1., size=(nrows, ncomp))
        V = np.random.normal(0., 1., size=(ncomp, ncols))
        data = np.dot(U, V)
        components = (U, V)

    elif data_str == 'clustering':
        U = np.random.multinomial(1, pi_crp, size=nrows)
        V = np.random.normal(0., 1., size=(ncomp, ncols))
        data = np.dot(U, V)
        components = (U, V)

    elif data_str == 'binary latent features':
        U = np.random.binomial(1, pi_ibp[nax,:], size=(nrows, ncomp))
        V = np.random.normal(0., 1., size=(ncomp, ncols))
        data = np.dot(U, V)
        components = (U, V)

    elif data_str == 'sparse coding':
        Z = np.random.normal(0., 1., size=(nrows, ncomp))
        U = np.random.normal(0., np.exp(Z))
        V = np.random.normal(0., 1., size=(ncomp, ncols))
        data = np.dot(U, V)
        components = (U, V)

    elif data_str == 's':
        Z = np.random.normal(0., 1., size=(nrows, ncols))
        U = np.random.normal(0., np.exp(Z))
        data = U

    elif data_str == 'dependent GSM':
        U_inner = np.random.normal(0., 1., size=(nrows, 1))
        V_inner = np.random.normal(0., 1., size=(1, ncomp))
        Z = np.random.normal(U_inner * V_inner, 1.)
        #Z = 2. * Z / np.sqrt(np.mean(Z**2))

        U = np.random.normal(0., np.exp(Z))
        V = np.random.normal(0., 1., size=(ncomp, ncols))
        data = np.dot(U, V)
        components = (U, V)

    elif data_str == 'co-clustering':
        U = np.random.multinomial(1, pi_crp, size=nrows)
        R = np.random.normal(0., 1., size=(ncomp, ncomp))
        V = np.random.multinomial(1, pi_crp, size=ncols).T
        data = np.dot(np.dot(U, R), V)
        components = (U, R, V)

    elif data_str == 'binary matrix factorization':
        U = np.random.binomial(1, pi_ibp[nax,:], size=(nrows, ncomp))
        R = np.random.normal(0., 1., size=(ncomp, ncomp))
        V = np.random.binomial(1, pi_ibp[nax,:], size=(ncols, ncomp)).T
        data = np.dot(np.dot(U, R), V)
        components = (U, R, V)

    elif data_str == 'MGB':
        U = np.random.multinomial(1, pi_crp, size=nrows)
        R = np.random.normal(0., 1., size=(ncomp, ncomp))
        V = np.random.binomial(1, pi_ibp[nax,:], size=(ncols, ncomp)).T
        data = np.dot(np.dot(U, R), V)
        components = (U, R, V)

    elif data_str == 'random_walk':
        data = generate_ar(nrows, ncols, 0.9)
        components = (data)

    elif data_str == 'linear dynamical system':
        U = generate_ar(nrows, ncomp, 0.9)
        V = np.random.normal(size=(ncomp, ncols))
        data = np.dot(U, V)
        components = (U, V)

    elif data_str == 'BCTF':
        temp1, (U1, V1) = generate_data('clustering', nrows, ncols, ncomp, True)
        F1 = np.random.normal(temp1, 1.)
        temp2, (U2, V2) = generate_data('clustering', nrows, ncols, ncomp, True)
        F2 = np.random.normal(temp2, 1.)
        data = np.dot(F1, F2.T)
        components = (U1, V1, F1, U2, V2, F2)

    elif data_str == 'G':
        data = np.zeros(shape=(nrows, ncols))

    data += np.random.normal(size=data.shape)
    data /= np.std(data)

    if transpose:
        data = data.T

    if return_components:
        return data, components
    else:
        return data

def pad(t, width=200):
    sp = t.shape
    t = np.expand_dims(t, axis=-1)
    mask = np.zeros_like(t)
    return np.concatenate(
        (np.pad(t, ((0, width - sp[0]), (0, width - sp[1]), (0, 0)), 'constant', constant_values=0.0),
         np.pad(mask, ((0, width - sp[0]), (0, width - sp[1]), (0, 0)), 'constant', constant_values=1.0)),
        axis=-1)

table = [
    ("low-rank", np_onehot(0, CANDIDATE_NUM), None),
    ("clustering", np_onehot(1, CANDIDATE_NUM), np_onehot(4, CANDIDATE_NUM)),
    ("binary latent features", np_onehot(2, CANDIDATE_NUM), np_onehot(5, CANDIDATE_NUM)),
    ("random_walk", np_onehot(3, CANDIDATE_NUM), np_onehot(6, CANDIDATE_NUM)),
    ("s", np_onehot(7, CANDIDATE_NUM), None),
    ("G", np_onehot(8, CANDIDATE_NUM), None)
]
TOTAL_CANDIDATE_NUM = 9
def main():
    config = tf.ConfigProto()#; config.gpu_options.allow_growth = TrueNone
    # Synthetic Data
    start_time = time.perf_counter()
    train_d = np.zeros([TRAIN_NUM * TOTAL_CANDIDATE_NUM, 200, 200, 2])
    train_l = np.zeros([TRAIN_NUM * TOTAL_CANDIDATE_NUM, CANDIDATE_NUM])
    val_d = np.zeros([VAL_NUM * TOTAL_CANDIDATE_NUM, 200, 200, 2])
    val_l = np.zeros([VAL_NUM * TOTAL_CANDIDATE_NUM, CANDIDATE_NUM])
    confusion = np.zeros([9, 9])
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    with tf.Session(config=config) as sess:
        c = Classifier()
        real = tf.placeholder(shape=[None, 200, 200, 2], dtype=tf.float32)
        c_out = tf.reshape(tf.nn.softmax(c(real)), [9])
        c_params = c.vars
        print(c_params)
        saver = tf.train.Saver(c_params)
        sess.run(tf.global_variables_initializer())
        if LOAD:
            saver.restore(sess, "saved_model/d")
        with tf.variable_scope("ground_truth", reuse=tf.AUTO_REUSE):
            for t in table:
                title, label, label_t = t
                for i in range(TRAIN_NUM):
                    h, w = max(3, int(np.random.uniform() * 200)), max(3, int(np.random.uniform() * 200))
                    r = min(int(np.random.uniform() * min(h - 3, w - 3)), 17) + 3
                    data = generate_data(title, h, w, r)
                    data -= np.mean(data)
                    data = pad(data)
                    confusion[np.argmax(label)] = confusion[np.argmax(label)] * 0.99 + sess.run(c_out, feed_dict={
                        real: np.expand_dims(data, axis=0)}) * 0.01
                    if label_t is not None:
                        data = generate_data(title, h, w, r)
                        data -= np.mean(data)
                        data = pad(data.T)
                        confusion[np.argmax(label_t)] = confusion[np.argmax(label_t)] * 0.99 + sess.run(c_out, feed_dict={
                            real: np.expand_dims(data, axis=0)}) * 0.01
                    if i % 1000 == 0:
                        print("processing data %s: (%d / %d)" % (title, i, TRAIN_NUM))
        print(confusion)
        conf_img = np.zeros([1, 900, 900, 1])
        for i in range(9):
            for j in range(9):
                conf_img[0, i * 100: i * 100 + 100, j * 100: j * 100 + 100, 0] = confusion[i, j] * 256.0
        save_img(conf_img, 1, file="data/results/confusion.png")
        conf_img = np.zeros([1, 900, 900, 1])
        for i in range(9):
            confusion[i, i] = 0.0
            confusion[i] /= np.sum(confusion[i])
        for i in range(9):
            for j in range(9):
                conf_img[0, i * 100: i * 100 + 100, j * 100: j * 100 + 100, 0] = confusion[i, j] * 256.0
        save_img(conf_img, 1, file="data/results/confusion-mutediag.png")
if __name__ == "__main__":
    main()
    log_file.close()
