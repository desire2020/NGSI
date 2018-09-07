import tensorflow as tf
import tensorflow.contrib.layers as tcl
import numpy as np
nax = np.newaxis
import tensorflow.contrib.autograph as autograph
import tensorflow.contrib.distributions as tfdist
import time
import gc


TRAIN_NUM = 5000
VAL_NUM = 100
CANDIDATE_NUM = 4
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


class Classifier(object):
    def __init__(self):
        self.name = 'metabayesian/d_net'

    def __call__(self, x, reuse=tf.AUTO_REUSE, split=False):
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE) as vs:
            x = tf.reshape(x, [-1, 200, 200, 2])
            conv0 = tcl.convolution2d(
                x, 32, [7, 7], [5, 5],
                activation_fn=tf.nn.relu
            )
            conv0 = tcl.convolution2d(
                conv0, 32, [3, 3], [1, 1],
                activation_fn=tf.nn.relu
            )
            conv0 = tcl.instance_norm(conv0)
            conv1 = tcl.convolution2d(
                conv0, 64, [6, 6], [4, 4],
                activation_fn=tf.nn.relu
            )
            conv1 = tcl.convolution2d(
                conv1, 64, [3, 3], [1, 1],
                activation_fn=tf.nn.relu
            )
            conv1 = tcl.instance_norm(conv1)
            conv2 = tcl.convolution2d(
                conv1, 128, [7, 7], [5, 5],
                activation_fn=tf.nn.relu
            )
            conv6 = tcl.flatten(conv2)
            fc2 = tcl.fully_connected(
                conv6, CANDIDATE_NUM,
                activation_fn=tf.identity
            )
            return fc2
    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]

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
    ("low-rank", np_onehot(0, CANDIDATE_NUM)),
    ("clustering", np_onehot(1, CANDIDATE_NUM)),
    ("binary latent features", np_onehot(2, CANDIDATE_NUM)),
    ("random_walk", np_onehot(3, CANDIDATE_NUM)),
    # ("co-clustering", np_onehot(4, CANDIDATE_NUM)),
    # ("binary matrix factorization", np_onehot(5, CANDIDATE_NUM)),
    # ("BCTF", np_onehot(6, CANDIDATE_NUM)),
    # ("sparse coding", np_onehot(7, CANDIDATE_NUM)),
    # ("dependent GSM", np_onehot(8, CANDIDATE_NUM)),
    # ("linear dynamical system", np_onehot(9, CANDIDATE_NUM)),
]

def main():
    config = tf.ConfigProto()#; config.gpu_options.allow_growth = True
    # Synthetic Data
    start_time = time.perf_counter()
    train_d = []
    train_l = []
    val_d = []
    val_l = []
    with tf.Session(config=config) as sess:
        oracle = {}
        c = Classifier()
        real = tf.placeholder(shape=[None, 200, 200, 2], dtype=tf.float32)
        real_l = tf.placeholder(shape=[None, CANDIDATE_NUM], dtype=tf.float32)
        c_out = c(real)
        c_loss = tf.reduce_mean(tf.reduce_sum(-tf.nn.log_softmax(c_out, axis=-1) * real_l, axis=-1))
        c_opt = tf.train.AdamOptimizer(2e-4, beta1=0.9, beta2=0.95)
        c_params = c.vars
        c_update = c_opt.minimize(c_loss, var_list=c_params)
        c_acc = tf.reduce_mean(
            (tf.cast(
                tf.equal(tf.argmax(c_out, axis=-1), tf.argmax(real_l, axis=-1)),
                tf.float32
            )))
        sess.run(tf.global_variables_initializer())
        with tf.variable_scope("ground_truth", reuse=tf.AUTO_REUSE):
            for t in table:
                title, label = t
                for i in range(TRAIN_NUM):
                    h, w = max(3, int(np.random.uniform() * 200)), max(3, int(np.random.uniform() * 200))
                    r = min(int(np.random.uniform() * min(h - 3, w - 3)), 17) + 3
                    data = generate_data(title, h, w, r)
                    data = pad(data)
                    train_d.append(data)
                    train_l.append(label)
                for i in range(VAL_NUM):
                    h, w = max(3, int(np.random.uniform() * 200)), max(3, int(np.random.uniform() * 200))
                    r = min(int(np.random.uniform() * min(h - 3, w - 3)), 17) + 3
                    data = generate_data(title, h, w, r)
                    data = pad(data)
                    val_d.append(data)
                    val_l.append(label)
        sess.run(tf.global_variables_initializer())
        val_d = np.reshape(val_d, [VAL_NUM * CANDIDATE_NUM, 200, 200, 2])
        val_l = np.reshape(val_l, [VAL_NUM * CANDIDATE_NUM, CANDIDATE_NUM])
        for i in range(1000000):
            if i % ((TRAIN_NUM * CANDIDATE_NUM) // 50) == 0:
                train_d = np.reshape(train_d, [TRAIN_NUM * CANDIDATE_NUM, 200, 200, 2])
                train_l = np.reshape(train_l, [TRAIN_NUM * CANDIDATE_NUM, CANDIDATE_NUM])
                shuffle_idx = list(range(TRAIN_NUM * CANDIDATE_NUM))
                np.random.shuffle(shuffle_idx)
                train_d = train_d[shuffle_idx]
                train_l = train_l[shuffle_idx]
                train_d = np.reshape(train_d, [-1, 50, 200, 200, 2])
                train_l = np.reshape(train_l, [-1, 50, CANDIDATE_NUM])
            data, lab = train_d[i % ((TRAIN_NUM * CANDIDATE_NUM) // 50)], train_l[i % ((TRAIN_NUM * CANDIDATE_NUM) // 50)]
            _, loss, train_acc = sess.run([c_update, c_loss, c_acc], feed_dict={real: data, real_l: lab})
            v_loss, test_acc = sess.run([c_loss, c_acc], feed_dict={real: val_d, real_l: val_l})
            print("iteration #%d, train_loss %f, val_loss %f, train acc %f, test acc %f" % (i, loss, v_loss, train_acc, test_acc))
if __name__ == "__main__":
    main()
    log_file.close()
