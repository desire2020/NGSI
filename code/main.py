import tensorflow as tf
import tensorflow.contrib.layers as tcl
import numpy as np
import tensorflow.contrib.autograph as autograph
import tensorflow.contrib.distributions as tfdist
from tensorflow.keras.backend import random_binomial
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
    "G": {"+@GGG", "+@MGG", "+@G'MG",
          "+@CGG", "+@CGG", "+@G'CG",
          "*eGG", "+@BGG", "+@G'BG"},
    "M": {"+@MGG", "B"},
    "B": {"+@BGG"}
}

BATCH_SIZE = 64

def GaussianSample(param):
    dev, shape = param
    return np.random.normal(scale=dev, size=shape)
def BernoulliSample(param):
    a, b, shape = param
    pi = np.random.beta(a, b, size=[shape[-1]])
    return np.random.binomial(1, pi, shape)
def MultinomialSample(param, verbose=False):
    alpha, shape = param
    if verbose:
        print(alpha)
    pi = np.random.dirichlet(alpha, 1).reshape(shape[1])
    ret = np.random.multinomial(1, pi, shape[0])
    return ret
def IntegrationMatrix(n): # Assuming C has to be nxn
    return np.cumsum(np.identity(n), axis=0)

def gaussian_sample(param):
    dev, shape = param
    return tf.expand_dims(dev, axis=0) * tf.random_normal([BATCH_SIZE] + shape)

def bernoulli_sample(param):
    anb, shape = param
    a, b = tf.unstack(anb)
    beta1 = tf.random_gamma([BATCH_SIZE] + shape, a)
    beta2 = tf.random_gamma([BATCH_SIZE] + shape, b)
    beta_sample = beta1 / (beta1 + beta2)
    return tf.cast(tf.random_uniform([BATCH_SIZE] + shape, 0.0, 1.0) < beta_sample, tf.float32)

def multinomial_sample(param):
    alpha, shape = param
    dirichlet = tf.random_gamma([BATCH_SIZE], alpha)
    dirichlet_sample = tf.log(dirichlet / tf.reduce_sum(dirichlet, axis=-1, keepdims=True))
    multinomial_sampled = tf.multinomial(dirichlet_sample, shape[0])
    return tf.one_hot(multinomial_sampled, shape[-1], 1.0, 0.0)

def intergration_matrix(n):
    return tf.tile(tf.expand_dims(tf.cumsum(tf.diag(tf.ones([n], dtype=tf.float32))), axis=0), [BATCH_SIZE, 1, 1])



def linear(t, name, width):
    in_w = t.get_shape().as_list()[-1]
    with tf.variable_scope(name):
        W = tf.get_variable("weight", initializer=tf.random_normal([in_w, width], stddev=0.02))
        b = tf.get_variable("bias", initializer=tf.zeros([width]))
    return t @ W + b


def generator(z):
    with tf.variable_scope("generator", reuse=tf.AUTO_REUSE):
        l0 = tf.nn.relu(linear(z, "layer0", 512))
        l1 = tf.nn.relu(linear(l0, "layer1", 512))
        l2 = tf.nn.relu(linear(l1, "layer2", 512))
        l3 = tf.nn.relu(linear(l2, "layer3", 512))
        o = linear(l3, "layer4", 256)
    return o


class Discriminator(object):
    def __init__(self):
        self.name = 'mnist/dcgan/d_net'

    def __call__(self, x, reuse=tf.AUTO_REUSE):
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE) as vs:
            x = tf.reshape(x, [-1, 64])
            fc0 = tcl.fully_connected(
                x, 2048,
                activation_fn=tf.nn.leaky_relu
            )
            fc1 = tcl.fully_connected(
                fc0, 2048,
                activation_fn=tf.nn.leaky_relu
            )
            fc2 = tcl.fully_connected(
                fc1, 2048,
                activation_fn=tf.nn.leaky_relu
            )
            fc3 = tcl.fully_connected(
                fc2, 2048,
                activation_fn=tf.nn.leaky_relu
            )
            fc4 = tcl.fully_connected(
                fc3, 1,
                activation_fn=tf.identity
            )
            return fc4

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name in var.name]

def edit_expr(expr, idx, target):
    i = 0
    last_i = 0
    count = 0
    while (count <= idx):
        if expr[i] in {"G", "M", "B", "C"}:
            count += 1
            last_i = i
        i += 1
    return expr[0:last_i] + target + expr[i + 1:]
def parse_param(expr, shape, rank=None):
    pp = [c for c in expr if c in {"G", "M", "C", "B"}]
    transposed = len([c for c in expr if c == "'"]) == 1
    p, t = None, None
    if rank is None:
        rank = shape[-1]
    shape0, shape1 = [shape[0], rank], [rank, shape[1]]
    if len(pp) == 1: # meaning this is "B"
        p = [tf.Variable(initial_value=[1.0, 1.0])]
        t = [bernoulli_sample(p + [shape])]
    elif len(pp) == 2: # meaning
        p = [tf.Variable(initial_value=tf.ones(shape0)), tf.Variable(initial_value=tf.ones(shape1))]
        t = [gaussian_sample((p[0], shape0)), gaussian_sample((p[1], shape1))]
    else:
        p = [None, None, None]
        t = [None, None, None]
        if pp[1] == "C":
            rank = shape[1]
            shape0, shape1 = [shape[0], rank], [rank, shape[1]]
        if pp[0] == "G":
            p[0] = tf.Variable(initial_value=tf.ones(shape0))
            t[0] = gaussian_sample((p[0], shape0))
        elif pp[0] == "M":
            p[0] = tf.Variable(initial_value=tf.ones([rank]))
            t[0] = multinomial_sample((p[0], shape0))
        elif pp[0] == "B":
            p[0] = tf.Variable(initial_value=[1.0, 1.0])
            t[0] = bernoulli_sample((p[0], shape0))
        else:
            p[0] = tf.Variable(initial_value=0.0) # useless parameter
            rank = shape[0]
            t[0] = intergration_matrix(shape[0])
            shape0, shape1 = [shape[0], rank], [rank, shape[1]]
        if transposed:
            shape1 = [shape[1], rank]
            if pp[1] == "G":
                p[1] = tf.Variable(initial_value=tf.ones(shape1))
                t[1] = gaussian_sample((p[1], shape1))
            elif pp[1] == "M":
                p[1] = tf.Variable(initial_value=tf.ones([rank]))
                t[1] = multinomial_sample((p[1], shape1))
            elif pp[1] == "B":
                p[1] = tf.Variable(initial_value=[1.0, 1.0])
                t[1] = bernoulli_sample((p[1], shape1))
            else:
                p[1] = tf.Variable(initial_value=0.0)  # useless parameter
                t[1] = intergration_matrix(rank)
        if pp[1] == "G":
            p[1] = tf.Variable(initial_value=tf.ones(shape1))
            t[1] = gaussian_sample((p[1], shape1))
        elif pp[1] == "M":
            p[1] = tf.Variable(initial_value=tf.ones([rank]))
            t[1] = multinomial_sample((p[1], shape1))
        elif pp[1] == "B":
            p[1] = tf.Variable(initial_value=[1.0, 1.0])
            t[1] = bernoulli_sample((p[1], shape1))
        else:
            p[1] = tf.Variable(initial_value=0.0)  # useless parameter
            t[1] = intergration_matrix(rank)
        p[2] = tf.Variable(initial_value=tf.ones(shape))
        t[2] = gaussian_sample((p[2], shape))
    return p, t

class Graph(object):
    def __init__(self):
        self.shape = [8, 8]
        self.real = tf.placeholder(shape=[64] + self.shape, dtype=tf.float32)
        self.fake = None
        self.sess = None
        self.params = [tf.Variable(initial_value=tf.ones(self.shape))]
        self.tensors = [tf.random_normal(shape=[64] + self.shape) * tf.reshape(self.params[0], [1] + self.shape)]
        self.grad_norm = [tf.Variable(0.0, trainable=False)]
        self.expr = "G"
        self.d = Discriminator()
        self.candidates = [] # list of pairs (self.tensors, self.expr, self.fake, self.converged_d_loss)
        self.real_score = tf.reduce_mean(self.d(self.real))
        self.d_loss = None
        self.g_loss = None
        self.current_eval = tf.Variable(0.0, trainable=False)
        self.update_grad_norm = None
        self.update_discriminator = None
        self.update_eval = None
        self.d_params = self.d.vars
        self.d_saver = tf.train.Saver(self.d_params)

    def update_model(self, idx, target):
        self.g_saver = tf.train.Saver(self.tensors)
        self.g_saver.save(self.sess, "saved_model/g_%s" % self.expr)
        self.push_tensors = self.tensors.copy()
        self.push_expr = self.expr
        self.push_fake = self.fake
        self.expr = edit_expr(self.expr, idx, target)
        p, t = parse_param(target, self.tensors[idx].get_shape())
        self.params = self.tensors[0:idx] + p + self.tensors[idx+1:]
        self.tensors = self.tensors[0:idx] + t + self.tensors[idx+1:]
        self.execute()
        self.d_saver.save(self.sess, "saved_model/d")
        self.candidates = []
    def execute(self):
        self.current_tensor_idx = 0
        def __parse(idx):
            if idx >= len(self.expr):
                print("PARSE ERROR")
            if self.expr[idx] == "+":
                param, _idx = __parse(idx + 1)
                _param, __idx = __parse(_idx)
                return param + _param, __idx
            elif self.expr[idx] == "@":
                param, _idx = __parse(idx + 1)
                _param, __idx = __parse(_idx)
                return param @ _param, __idx
            elif self.expr[idx] == "'":
                param, _idx = __parse(idx + 1)
                return tf.transpose(param, [0, 2, 1]), _idx
            elif self.expr[idx] == "e":
                param, _idx = __parse(idx + 1)
                return tf.exp(param), _idx
            else:
                self.current_tensor_idx += 1
                return self.tensors[self.current_tensor_idx - 1]
        self.fake = __parse(0)[0]

    def rewind_model(self):
        self.candidates.append((self.tensors, self.expr, self.fake, self.sess.run(self.current_eval)))
        self.tensors = self.push_tensors
        self.expr = self.push_expr
        self.fake = self.push_fake
        self.g_saver = tf.train.Saver(self.tensors)
        self.g_saver.restore(self.sess, "saved_model/g_%s" % self.expr)
        self.d_saver.restore(self.sess, "saved_model/d")

def main():
    config = tf.ConfigProto(); config.gpu_options.allow_growth = True; sess = tf.Session(config=config)
    synthetic = [(np.array([1.0, 1.0, 1.0]), [8, 3]),
                 ((np.array([[0.1], [0.1], [0.1], [0.1], [0.1], [0.1], [0.1], [0.1]]) @ np.array([[0.3, 0.8, 0.3]])).transpose(), [3, 8]),
                 (np.ones([8, 8]) * 0.01, [8, 8])]
    model = Graph()
    model.sess = sess
    sess.run(tf.global_variables_initializer())
    for i in range(1000000):
        D_loss = []
        for _ in range(10):
            real = []
            for _ in range(64):
                real.append(MultinomialSample(synthetic[0]))# @ GaussianSample(synthetic[1]) + GaussianSample(synthetic[2]))
            real = np.stack(real)
            d_loss, _ = sess.run([model.d_loss, model.update], feed_dict={model.real: real})
            D_loss.append(d_loss)
        D_loss = np.mean(D_loss)
        sess.run(model.update_dist)
        if i % 100 == 0:
            print("d_loss at iteration #%d: %f" % (i, D_loss))
            print("now alpha is:")
            for v in model.alphas:
                print(v)

if __name__ == "__main__":
    main()
