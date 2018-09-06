import tensorflow as tf
import tensorflow.contrib.layers as tcl
import numpy as np
import tensorflow.contrib.autograph as autograph
import tensorflow.contrib.distributions as tfdist
import time
import gc
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
          "+@BGG", "+@G'BG"], #, "+@CGG", "+@G'CG"],
    "M": ["+@MGG", "B"],
    "B": ["+@BGG"]
}

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
    dev = tf.reshape(dev, [1, -1])
    return dev * tf.random_normal(shape)

def bernoulli_sample(param):
    anb, shape = param
    a, b = tf.unstack(anb)
    beta1 = tf.random_gamma(shape, a)
    beta2 = tf.random_gamma(shape, b)
    beta_sample = beta1 / (beta1 + beta2)
    t = tf.cast(tf.random_uniform(shape, 0.0, 1.0) < beta_sample, tf.float32)
    g = tf.tile(tf.reshape(a / (a + b), [1, 1]), shape)
    return tf.stop_gradient(t) + g - tf.stop_gradient(g)

def multinomial_sample(param):
    alpha, shape = param
    dirichlet = tf.random_gamma([1], alpha)
    dirichlet_sample = tf.log(dirichlet / tf.reduce_sum(dirichlet, axis=-1, keepdims=True))
    multinomial_sampled = tf.multinomial(dirichlet_sample, shape[0])
    t = tf.reshape(tf.one_hot(multinomial_sampled, shape[-1], 1.0, 0.0), shape)
    g = tf.tile(tf.reshape(alpha, [1, -1]), [shape[0], 1])
    return tf.stop_gradient(t) + g - tf.stop_gradient(g)

def intergration_matrix(n):
    return tf.cumsum(tf.diag(tf.ones([n], dtype=tf.float32)))



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
        self.name = 'metabayesian/d_net'

    def __call__(self, x, reuse=tf.AUTO_REUSE, split=False):
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE) as vs:
            x = tf.reshape(x, [-1, 1, 1, 200])
            conv0 = tcl.convolution2d(
                x, 20, [1, 1], [1, 1],
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=tf.nn.leaky_relu
            )
            conv1 = tcl.convolution2d(
                conv0, 20, [1, 1], [1, 1],
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=tf.nn.leaky_relu
            )
            conv6 = tcl.convolution2d(
                conv1, 20, [1, 1], [1, 1],
                weights_initializer=tf.random_normal_initializer(stddev=0.02),
                activation_fn=tf.nn.leaky_relu
            )
            conv6 = tcl.flatten(conv6)
            fc2 = tcl.fully_connected(
                conv6, 1,
                activation_fn=tf.identity
            )
            if split:
                return fc2[0:100, :], fc2[100:, :]
            else:
                return fc2
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
    return expr[0:last_i] + target + expr[i:]
def parse_param(expr, shape, rank=None, initializer=0.0):
    pp = [c for c in expr if c in {"G", "M", "C", "B"}]
    transposed = len([c for c in expr if c == "'"]) == 1
    p, t = None, None
    if rank is None:
        rank = shape[-1]
    shape0, shape1 = [shape[0], rank], [rank, shape[1]]
    if len(pp) == 1: # meaning this is "B"
        if pp[0] == "B":
            p = [tf.Variable(initial_value=[1.0, 1.0])]
            t = [bernoulli_sample(p + [shape])]
        else:
            p = [tf.Variable(initial_value=tf.ones([shape[-1]]) * initializer)]
            t = [gaussian_sample((p[0], shape))]
    elif len(pp) == 2: # meaning
        p = [tf.Variable(initial_value=tf.ones([shape[-1]]) * initializer),
             tf.Variable(initial_value=tf.ones([shape[-1]]) * initializer)]
        t = [gaussian_sample((p[0], shape)), gaussian_sample((p[1], shape))]
    else:
        p = [None, None, None]
        t = [None, None, None]
        if pp[1] == "C":
            rank = shape[1]
            shape0, shape1 = [shape[0], rank], [rank, shape[1]]
        if pp[0] == "G":
            p[0] = tf.Variable(initial_value=tf.ones([shape0[-1]]) * initializer)
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
                p[1] = tf.Variable(initial_value=tf.ones([shape1[-1]]) * initializer)
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
            p[1] = tf.Variable(initial_value=tf.ones([shape1[-1]]) * initializer)
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
        p[2] = tf.Variable(initial_value=tf.ones([shape[-1]]) * initializer)
        t[2] = gaussian_sample((p[2], shape))
    return p, t

class Graph(object):
    def __init__(self):
        self.shape = [200, 200]
        self.real = tf.placeholder(shape=self.shape, dtype=tf.float32)
        self.fake = None
        self.sess = None
        self.params = [tf.Variable(initial_value=tf.ones(self.shape))]
        self.tensors = [tf.random_normal(shape=self.shape) * tf.reshape(self.params[0], self.shape)]
        self.component_biases = [tf.Variable(0.0, trainable=False)]
        self.expr = "G"
        self.fake = self.tensors[0]
        self.d = Discriminator()
        self.candidates = [] # list of pairs (self.tensors, self.expr, self.fake, self.converged_d_loss)
        self.real_score, self.val_score = self.d(self.real, split=True)
        self.real_score, self.val_score = tf.reduce_mean(self.real_score), tf.reduce_mean(self.val_score)
        self.d_loss = None
        self.g_loss = None
        self.g_update = None
        self.parent_eval = tf.Variable(0.0, trainable=False)
        self.current_eval = tf.Variable(0.0, trainable=False)
        self.update_grad_norm = None
        self.update_discriminator = None
        self.update_eval = None
        self.d_params = self.d.vars
        self.d_opt = tf.train.AdamOptimizer(1e-3, beta1=0.5, beta2=0.9)
        self.g_opt = tf.train.AdamOptimizer(1e-3, beta1=0.5, beta2=0.9)
        self.d_saver = tf.train.Saver(self.d_params)

    def update_model(self, idx, target, rank=10, initializer=0.0):
        self.g_saver = tf.train.Saver(self.params)
        self.g_saver.save(self.sess, "saved_model/g_%s" % self.expr)
        self.push_tensors = self.tensors.copy()
        self.push_expr = self.expr
        self.push_params = self.params.copy()
        self.push_fake = self.fake
        self.push_d_loss = self.d_loss
        self.push_g_loss = self.g_loss
        self.push_g_update = self.g_update
        self.push_cb = self.component_biases.copy()
        self.push_ugn = self.update_grad_norm
        self.push_ue = self.update_eval
        self.push_ud = self.update_discriminator
        self.expr = edit_expr(self.expr, idx, target)
        p, t = parse_param(target, self.tensors[idx].get_shape().as_list(), rank, initializer)
        self.sess.run(tf.variables_initializer(p))
        self.params = self.params[0:idx] + p + self.params[idx+1:]
        self.tensors = self.tensors[0:idx] + t + self.tensors[idx+1:]
        self.component_biases = [tf.Variable(0.0, trainable=False) for _ in self.tensors]
        self.sess.run(tf.variables_initializer(self.component_biases))
        self.execute()
        self.fake_score = tf.reduce_mean(self.d(self.fake))
        self.d_loss = self.real_score - self.fake_score
        self.sess.run(self.parent_eval.assign(self.current_eval))
        self.sess.run(self.current_eval.assign(0.0))
        sample_expand = 8
        hybrid = tf.random_uniform(shape=[200 * sample_expand, 1], minval=0.0, maxval=1.0)
        hybrid = tf.tile(self.real, [sample_expand, 1]) * hybrid + tf.tile(self.fake, [sample_expand, 1]) * (1.0 - hybrid)
        gp = 50.0 * tf.reduce_mean(
            tf.nn.relu(tf.norm(tf.reshape(tf.gradients(self.d(hybrid), [hybrid])[0], [-1, 200]), axis=-1) - 1.0) ** 2)
        self.g_loss = self.fake_score
        self.g_update = self.g_opt.minimize(self.g_loss, var_list=self.params)
        self.d_saver.save(self.sess, "saved_model/d")
        self.grad_norm = tf.gradients(self.g_loss, self.tensors)
        self.update_grad_norm = [self.component_biases[i].assign(self.component_biases[i] * 0.99 + tf.reduce_mean(self.grad_norm[i] ** 2)) * 0.01 for i in range(len(self.grad_norm))]
        self.update_eval = self.current_eval.assign(self.current_eval * 0.99 + (self.val_score - self.fake_score) * 0.01)
        self.update_discriminator = self.d_opt.minimize(self.d_loss + gp, var_list=self.d_params)
    def execute(self):
        self.current_tensor_idx = 0
        def __parse(idx):
            # print("at idx #%d of expr \"%s\"" % (idx, self.expr))
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
            elif self.expr[idx] == "*":
                param, _idx = __parse(idx + 1)
                _param, __idx = __parse(_idx)
                return param * _param, __idx
            elif self.expr[idx] == "'":
                param, _idx = __parse(idx + 1)
                return tf.transpose(param, [1, 0]), _idx
            elif self.expr[idx] == "e":
                param, _idx = __parse(idx + 1)
                return tf.exp(param), _idx
            else:
                self.current_tensor_idx += 1
                return self.tensors[self.current_tensor_idx - 1], idx + 1
        self.fake = __parse(0)[0]
    def train_d(self, real_x):
        self.sess.run([self.update_discriminator, self.update_eval, self.update_grad_norm], feed_dict={self.real: real_x})
    def train_g(self):
        self.sess.run(self.g_update)
    def generate(self):
        return self.sess.run(self.fake)

    def rewind_model(self):
        self.candidates.append((self.tensors.copy(), self.params.copy(), self.expr, self.fake,
                                self.d_loss, self.g_loss,
                                self.component_biases.copy(), self.update_grad_norm,
                                self.update_eval, self.update_discriminator,
                                self.sess.run(self.current_eval), self.g_update))
        self.tensors = self.push_tensors
        self.params = self.push_params
        self.expr = self.push_expr
        self.fake = self.push_fake
        self.d_loss = self.push_d_loss
        self.g_update = self.push_g_update
        self.g_loss = self.push_g_loss
        self.component_biases = self.push_cb
        self.update_grad_norm = self.push_ugn
        self.update_eval = self.push_ue
        self.update_discriminator = self.push_ud
        self.sess.run(self.current_eval.assign(self.parent_eval))
        self.g_saver = tf.train.Saver(self.params)
        self.g_saver.restore(self.sess, "saved_model/g_%s" % self.expr)
        self.d_saver.restore(self.sess, "saved_model/d")

    def ensure_update(self, idx, baseline):
        self.tensors, self.params, self.expr, self.fake,\
        self.d_loss, self.g_loss,\
        self.component_biases, self.update_grad_norm,\
        self.update_eval, self.update_discriminator, _, self.g_update = self.candidates[idx]
        self.candidates = []
        self.sess.run(self.current_eval.assign(baseline))
# Synthetic Data Table
table = [
    ("low-rank", [(0, "+@G'GG", 10)]),
    ("clustering", [(0, "+@MGG", 10)]),
    ("binary latent features", [(0, "+@BGG", 10)]),
    ("co-clustering", [(0, "+@MGG", 10), (1, "+@G'MG", 10)]),
    ("binary matrix factorization", [(0, "+@G'BG", 10), (0, "+@BGG", 10)]),
    ("BCTF", [(0, "+@GGG", 10), (0, "+@MGG", 10), (1, "+@MGG", 10)]),
    ("sparse coding", [(0, "+@G'GG", 10), (0, "*eGG", 10)]),
    ("dependent GSM", [(0, "+@G'GG", 10), (0, "*eGG", 10), (0, "*@G'GG", 10)]),
    ("random_walk", [(0, "+@CGG", 10)]),
    ("linear dynamical system", [(0, "+@CGG", 10), (0, "+@CGG", 10)]),
]

def check_and_handle_uninit(sess):
    vl = tf.global_variables()
    p = [tf.is_variable_initialized(v) for v in vl]
    p = sess.run(p)
    v = [vl[i] for i in range(len(vl)) if not p[i]]
    sess.run(tf.variables_initializer(v))

ESTIMATION_ITERATIONS = 10001

log_file = open("saved_model/log.txt", "w")
def main():
    config = tf.ConfigProto()#; config.gpu_options.allow_growth = True
    # Synthetic Data
    for t in table:
        title, sequence = t
        tf.reset_default_graph()
        start_time = time.perf_counter()
        with tf.Session(config=config) as sess:
            with tf.variable_scope("ground_truth", reuse=tf.AUTO_REUSE):
                oracle = Graph()
                oracle.sess = sess
                sess.run(tf.global_variables_initializer())
                for op in sequence:
                    idx, ops, rank = op
                    oracle.update_model(idx, ops, rank, 1.0)
            data = oracle.generate()
            print("oracle data: %s" % title)
            print("oracle expr:%s" % oracle.expr)
            model = Graph()
            model.sess = sess
            check_and_handle_uninit(sess)
            model.update_model(0, "G")
            check_and_handle_uninit(sess)
            for i in range(ESTIMATION_ITERATIONS):
                for d_idx in range(3):
                    model.train_d(data)
                if i % 1000 == 0:
                    print("model %s, current eval at iteration #%d: %f" % (model.expr, i, sess.run(model.current_eval)))
                    log_file.write(
                        "model %s, current eval at iteration #%d: %f\n" % (model.expr, i, sess.run(model.current_eval)))
                model.train_g()
            # model.update_model(0, "+@GGG", 10)
            # model.update_model(0, "+@GGG", 1)
            for depth in range(3):
                idx = sess.run(tf.argmax(model.component_biases))
                print("Detected biased component: %d of \"%s\"" % (idx, model.expr))
                origin = [c for c in model.expr if c in {"G", "M", "B", "C"}][idx]
                parent = model.expr
                for ops in induction_rule[origin]:
                    model.update_model(idx, ops)
                    sess.run(tf.variables_initializer(model.d_params))
                    check_and_handle_uninit(sess)
                    print("testing model %s" % model.expr)
                    for i in range(ESTIMATION_ITERATIONS):
                        for d_idx in range(3):
                            model.train_d(data)
                        if i % 1000 == 0:
                            print("model %s, current eval at iteration #%d: %f, parent eval %f" % (model.expr, i, sess.run(model.current_eval), sess.run(model.parent_eval)))
                            log_file.write("model %s, current eval at iteration #%d: %f, parent eval %f\n" % (model.expr, i, sess.run(model.current_eval), sess.run(model.parent_eval)))
                       #  if i > 10 and sess.run(tf.nn.relu(-model.current_eval) > -3.0 * model.parent_eval):
                          #  print("abandoned.")
                          #  break
                        model.train_g()
                    model.rewind_model()
                baseline = sess.run(model.parent_eval)
                print("parent model \"%s\", scored %f" % (parent, baseline))
                selected = -1
                for i in range(len(model.candidates)):
                    c = model.candidates[i]
                    print("model: \"%s\", scored %f" % (c[2], c[-2]))
                    if c[-2] > baseline:
                        print("this is the best over all currently checked model.")
                        selected = i
                        baseline = c[-2]
                if selected == -1:
                    end_time = time.perf_counter() - start_time
                    print("Final result: model %s for data %s, time %s s" % (parent, title, end_time))
                    log_file.write("Final result: model %s for data %s, time %s s\n" % (parent, title, end_time))
                    gc.collect()
                    break
                else:
                    end_time = time.perf_counter() - start_time
                    model.ensure_update(selected, baseline)
                    print("for data %s, model now is switched to \"%s\", time %f s." % (title, model.expr, end_time))
                    log_file.write("for data %s, model now is switched to \"%s\", time %f s.\n" % (title, model.expr, end_time))
                    gc.collect()
if __name__ == "__main__":
    main()
    log_file.close()
