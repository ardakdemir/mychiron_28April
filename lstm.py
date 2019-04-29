from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn.python.ops.core_rnn_cell import RNNCell
from keras.layers import RNN,LSTM
from keras import *

class LSTMCell(RNNCell):
    '''Vanilla LSTM implemented with same initializers as BN-LSTM'''

    def __init__(self, num_units):
        self.num_units = num_units

    @property
    def state_size(self):
        return (self.num_units, self.num_units)

    @property
    def output_size(self):
        return self.num_units

    def __call__(self, x, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            c, h = state

            # Keep W_xh and W_hh separate here as well to reuse initialization methods
            x_size = x.get_shape().as_list()[1]
            W_xh = tf.get_variable('W_xh',
                                   [x_size, 4 * self.num_units],
                                   initializer=orthogonal_initializer())
            W_hh = tf.get_variable('W_hh',
                                   [self.num_units, 4 * self.num_units],
                                   initializer=bn_lstm_identity_initializer(0.95))
            bias = tf.get_variable('bias', [4 * self.num_units])

            # hidden = tf.matmul(x, W_xh) + tf.matmul(h, W_hh) + bias
            # improve speed by concat.
            concat = tf.concat(axis=1, values=[x, h])
            W_both = tf.concat(axis=0, values=[W_xh, W_hh])
            hidden = tf.matmul(concat, W_both) + bias

            i, j, f, o = tf.split(axis=1, num_or_size_splits=4, value=hidden)

            new_c = c * tf.sigmoid(f) + tf.sigmoid(i) * tf.tanh(j)
            new_h = tf.tanh(new_c) * tf.sigmoid(o)

            return new_h, (new_c, new_h)

class LSTMBN(LSTM):
    '''Long-Short Term Memory unit - Hochreiter 1997.
    with Batch Normalization support
    For a step-by-step description of the algorithm, see
    [this tutorial](http://deeplearning.net/tutorial/lstm.html).
    # Arguments
        output_dim: dimension of the internal projections and the final output.
        init: weight initialization function.
            Can be the name of an existing function (str),
            or a Theano function (see: [initializers](../initializers.md)).
        inner_init: initialization function of the inner cells.
        forget_bias_init: initialization function for the bias of the forget gate.
            [Jozefowicz et al.](http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)
            recommend initializing with ones.
        activation: activation function.
            Can be the name of an existing function (str),
            or a Theano function (see: [activations](../activations.md)).
        inner_activation: activation function for the inner cells.
        W_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the input weights matrices.
        U_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the recurrent weights matrices.
        b_regularizer: instance of [WeightRegularizer](../regularizers.md),
            applied to the bias.
        dropout_W: float between 0 and 1. Fraction of the input units to drop for input gates.
        dropout_U: float between 0 and 1. Fraction of the input units to drop for recurrent connections.
        batch_norm: bool (default False)
            Perform batch normalization as described in
            [Recurrent Batch Normalization](http://arxiv.org/abs/1603.09025)
             we the simplification of using the same BN parameters on all steps.
        gamma_init: float (default 0.1)
             initalization value for all gammas used in batch normalization.
    # References
        - [Long short-term memory](http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf) (original 1997 paper)
        - [Learning to forget: Continual prediction with LSTM](http://www.mitpressjournals.org/doi/pdf/10.1162/089976600300015015)
        - [Supervised sequence labelling with recurrent neural networks](http://www.cs.toronto.edu/~graves/preprint.pdf)
        - [A Theoretically Grounded Application of Dropout in Recurrent Neural Networks](http://arxiv.org/abs/1512.05287)
        - [Recurrent Batch Normalization](http://arxiv.org/abs/1603.09025)
    '''
    def __init__(self, output_dim,
                 init='glorot_uniform', inner_init='orthogonal',
                 forget_bias_init='one', activation='tanh',
                 inner_activation='hard_sigmoid',
                 W_regularizer=None, U_regularizer=None, b_regularizer=None,
                 dropout_W=0., dropout_U=0.,
                 batch_norm=False, gamma_init=0.1, **kwargs):
        super(LSTM, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.init = initializers.get(init)
        self.inner_init = initializers.get(inner_init)
        self.forget_bias_init = initializers.get(forget_bias_init)
        self.activation = activations.get(activation)
        self.inner_activation = activations.get(inner_activation)
        self.W_regularizer = regularizers.get(W_regularizer)
        self.U_regularizer = regularizers.get(U_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.dropout_W, self.dropout_U = dropout_W, dropout_U
        self.batch_norm = batch_norm
        if batch_norm:
            def gamma_init_func(shape, name=None, c=gamma_init):
                return K.variable(np.ones(shape) * c, name=name)
            self.gamma_init = gamma_init_func
            self.beta_init = initializers.get('zero')
            self.momentum = 0.9
            self.epsilon = 1e-6
            self.uses_learning_phase = True
        if self.dropout_W or self.dropout_U:
            self.uses_learning_phase = True


    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        input_dim = input_shape[2]
        self.input_dim = input_dim

        if self.batch_norm:
            shape = (self.output_dim,)
            shape1 = (self.output_dim,)
            self.gammas = {}
            self.betas = {}
            self.running_mean = {}
            self.running_std = {}
            # BN is applied in 3 inputs/outputs (fields) of the cell
            for fld in ['recurrent', 'input', 'output']:
                gammas = {}
                betas = {}
                running_mean = {}
                running_std = {}
                # each of the fields affects 4 locations inside the cell
                # (except output)
                # each location has its own BN
                for slc in ['i', 'f', 'c', 'o']:
                    running_mean[slc] = K.zeros(shape1,
                                                name='{}_running_mean_{}_{}'.format(
                                                    self.name, fld, slc))
                    running_std[slc] = K.ones(shape1,
                                              name='{}_running_std_{}_{}'.format(
                                                  self.name, fld, slc))
                    gammas[slc] = self.gamma_init(shape,
                                                  name='{}_gamma_{}_{}'.format(
                                                      self.name, fld, slc))
                    if fld == 'output':
                        betas[slc] = self.beta_init(shape,
                                                    name='{}_beta_{}_{}'.format(
                                                        self.name, fld, slc))
                        break  # output has just one slice

                self.gammas[fld] = gammas
                self.betas[fld] = betas
                self.running_mean[fld] = running_mean
                self.running_std[fld] = running_std

        if self.stateful:
            self.reset_states()
        else:
            # initial states: 2 all-zero tensors of shape (output_dim)
            self.states = [None, None]

        self.W_i = self.init((input_dim, self.output_dim),
                             name='{}_W_i'.format(self.name))
        self.U_i = self.inner_init((self.output_dim, self.output_dim),
                                   name='{}_U_i'.format(self.name))
        self.b_i = K.zeros((self.output_dim,), name='{}_b_i'.format(self.name))

        self.W_f = self.init((input_dim, self.output_dim),
                             name='{}_W_f'.format(self.name))
        self.U_f = self.inner_init((self.output_dim, self.output_dim),
                                   name='{}_U_f'.format(self.name))
        self.b_f = self.forget_bias_init((self.output_dim,),
                                         name='{}_b_f'.format(self.name))

        self.W_c = self.init((input_dim, self.output_dim),
                             name='{}_W_c'.format(self.name))
        self.U_c = self.inner_init((self.output_dim, self.output_dim),
                                   name='{}_U_c'.format(self.name))
        self.b_c = K.zeros((self.output_dim,), name='{}_b_c'.format(self.name))

        self.W_o = self.init((input_dim, self.output_dim),
                             name='{}_W_o'.format(self.name))
        self.U_o = self.inner_init((self.output_dim, self.output_dim),
                                   name='{}_U_o'.format(self.name))
        self.b_o = K.zeros((self.output_dim,), name='{}_b_o'.format(self.name))

        self.regularizers = []
        if self.W_regularizer:
            self.W_regularizer.set_param(K.concatenate([self.W_i,
                                                        self.W_f,
                                                        self.W_c,
                                                        self.W_o]))
            self.regularizers.append(self.W_regularizer)
        if self.U_regularizer:
            self.U_regularizer.set_param(K.concatenate([self.U_i,
                                                        self.U_f,
                                                        self.U_c,
                                                        self.U_o]))
            self.regularizers.append(self.U_regularizer)
        if self.b_regularizer:
            self.b_regularizer.set_param(K.concatenate([self.b_i,
                                                        self.b_f,
                                                        self.b_c,
                                                        self.b_o]))
            self.regularizers.append(self.b_regularizer)

        self.trainable_weights = [self.W_i, self.U_i, self.b_i,
                                  self.W_c, self.U_c, self.b_c,
                                  self.W_f, self.U_f, self.b_f,
                                  self.W_o, self.U_o, self.b_o]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

        if self.batch_norm:
            self.non_trainable_weights = []
            for fld in ['recurrent', 'input', 'output']:
                self.trainable_weights += self.gammas[fld].values() + self.betas[fld].values()
                self.non_trainable_weights += (self.running_mean[fld].values() +
                                               self.running_std[fld].values())


    def add_bn_to_states(self, states, running_mean, running_std):
        if not self.batch_norm:
            return
        for fld in ['recurrent', 'input', 'output']:
            for slc in ['i', 'f', 'c', 'o']:
                states.append(K.replace_row(self.padding,
                                            running_mean[fld][slc]))
                states.append(K.replace_row(self.padding,
                                            running_std[fld][slc]))
                if fld == 'output':
                    break

    def reset_states(self):
        assert self.stateful, 'Layer must be stateful.'
        input_shape = self.input_spec[0].shape
        if not input_shape[0]:
            raise Exception('If a RNN is stateful, a complete ' +
                            'input_shape must be provided (including batch size).')
        if hasattr(self, 'states'):
            K.set_value(self.states[0],
                        np.zeros((input_shape[0], self.output_dim)))
            K.set_value(self.states[1],
                        np.zeros((input_shape[0], self.output_dim)))
        else:
            self.states = [K.zeros((input_shape[0], self.output_dim)),
                           K.zeros((input_shape[0], self.output_dim))]
            if self.batch_norm:
                self.padding = K.zeros((input_shape[0], self.output_dim))
                self.add_bn_to_states(self.states, self.running_mean, self.running_std)

    def get_initial_states(self, x):
        if not self.batch_norm:
            return super(LSTM,self).get_initial_states(x)

        # build an all-zero tensor of shape (samples, output_dim)
        initial_state = K.zeros_like(x)  # (samples, timesteps, input_dim)
        initial_state = K.sum(initial_state, axis=1)  # (samples, input_dim)
        reducer = K.zeros((self.input_dim, self.output_dim))
        initial_state = K.dot(initial_state, reducer)  # (samples, output_dim)
        initial_states = [initial_state for _ in range(len(self.states))]
        if self.batch_norm:
            self.padding = initial_state
            self.add_bn_to_states(initial_states, self.running_mean, self.running_std)
        return initial_states

    def preprocess_input(self, x):
        if self.consume_less == 'cpu':
            if 0 < self.dropout_W < 1:
                dropout = self.dropout_W
            else:
                dropout = 0
            input_shape = self.input_spec[0].shape
            input_dim = input_shape[2]
            timesteps = input_shape[1]

            # bias is added inside step (after doing BN)
            x_i = time_distributed_dense(x, self.W_i, None, dropout,
                                         input_dim, self.output_dim, timesteps)
            x_f = time_distributed_dense(x, self.W_f, None, dropout,
                                         input_dim, self.output_dim, timesteps)
            x_c = time_distributed_dense(x, self.W_c, None, dropout,
                                         input_dim, self.output_dim, timesteps)
            x_o = time_distributed_dense(x, self.W_o, None, dropout,
                                         input_dim, self.output_dim, timesteps)
            return K.concatenate([x_i, x_f, x_c, x_o], axis=2)
        else:
            return x

    def bn(self, X, fld, slc='i'):
        if not self.batch_norm:
            return X
        gamma = self.gammas[fld][slc]
        # recurrent and input fields dont have beta
        beta = self.betas[fld].get(slc)
        axis = -1  # axis along which to normalize (we have mode=0)

        input_shape = (self.input_spec[0].shape[0], self.output_dim)
        reduction_axes = list(range(len(input_shape)))
        del reduction_axes[axis]
        broadcast_shape = [1] * len(input_shape)
        broadcast_shape[axis] = input_shape[axis]

        # case: train mode (uses stats of the current batch)
        m = K.mean(X, axis=reduction_axes)
        brodcast_m = K.reshape(m, broadcast_shape)
        std = K.mean(K.square(X - brodcast_m) + self.epsilon,
                     axis=reduction_axes)
        std = K.sqrt(std)
        brodcast_std = K.reshape(std, broadcast_shape)
        mean_update = (self.momentum * self.step_running_mean[fld][slc] +
                       (1 - self.momentum) * m)
        std_update = (self.momentum * self.step_running_std[fld][slc] +
                      (1 - self.momentum) * std)

        X_normed = (X - brodcast_m) / (brodcast_std + self.epsilon)

        # case: test mode (uses running averages)
        brodcast_running_mean = K.reshape(self.step_running_mean[fld][slc], broadcast_shape)
        brodcast_running_std = K.reshape(self.step_running_std[fld][slc],
                                 broadcast_shape)
        X_normed_running = ((X - brodcast_running_mean) /
                        (brodcast_running_std + self.epsilon))

        # pick the normalized form of x corresponding to the training phase
        X_normed = K.in_train_phase(X_normed, X_normed_running)

        out = K.reshape(gamma, broadcast_shape) * X_normed
        if beta is not None:
            out += K.reshape(beta, broadcast_shape)

        self.out_step_running_mean[fld][slc] = mean_update
        self.out_step_running_std[fld][slc] = std_update
        return out

    def step(self, x, states):
        # unpack states to its variables
        i = 0
        h_tm1 = states[i] ; i += 1
        c_tm1 = states[i] ; i += 1
        if self.batch_norm:
            self.step_running_mean = {}
            self.step_running_std = {}
            self.out_step_running_mean = {}
            self.out_step_running_std = {}
            for fld in ['recurrent', 'input', 'output']:
                self.step_running_mean[fld] = {}
                self.step_running_std[fld] = {}
                self.out_step_running_mean[fld] = {}
                self.out_step_running_std[fld] = {}
                for slc in ['i', 'f', 'c', 'o']:
                    self.step_running_mean[fld][slc] = states[i][0,:] ; i += 1
                    self.step_running_std[fld][slc] = states[i][0,:] ; i += 1
                    if fld == 'output':
                        break
        # unpack constants
        B_U = states[i] ; i += 1
        B_W = states[i] ; i += 1

        if self.consume_less == 'cpu':
            x_i = x[:, :self.output_dim]
            x_f = x[:, self.output_dim: 2 * self.output_dim]
            x_c = x[:, 2 * self.output_dim: 3 * self.output_dim]
            x_o = x[:, 3 * self.output_dim:]
        else:
            x_i = K.dot(x * B_W[0], self.W_i) + self.b_i
            x_f = K.dot(x * B_W[1], self.W_f) + self.b_f
            x_c = K.dot(x * B_W[2], self.W_c) + self.b_c
            x_o = K.dot(x * B_W[3], self.W_o) + self.b_o

        i = self.inner_activation(
            self.bn(x_i,'input','i') +
            self.bn(K.dot(h_tm1 * B_U[0], self.U_i), 'recurrent','i') +
            self.b_i)
        f = self.inner_activation(
            self.bn(x_f,'input','f') +
            self.bn(K.dot(h_tm1 * B_U[1], self.U_f),'recurrent','f') +
            self.b_f)
        c = f * c_tm1 + i * self.activation(
            self.bn(x_c,'input','c') +
            self.bn(K.dot(h_tm1 * B_U[2], self.U_c),'recurrent','c') +
            self.b_c)
        o = self.inner_activation(
            self.bn(x_o,'input','o') +
            self.bn(K.dot(h_tm1 * B_U[3], self.U_o),'recurrent','o') +
            self.b_o)

        h = o * self.activation(self.bn(c, 'output'))

        out_states = [h, c]
        if self.batch_norm:
            self.add_bn_to_states(out_states,
                                  self.out_step_running_mean,
                                  self.out_step_running_std)
            del self.step_running_mean
            del self.step_running_std
            del self.out_step_running_mean
            del self.out_step_running_std
        return h, out_states

    def get_constants(self, x):
        constants = []
        if 0 < self.dropout_U < 1:
            ones = K.ones_like(K.reshape(x[:, 0, 0], (-1, 1)))
            ones = K.concatenate([ones] * self.output_dim, 1)
            B_U = [K.in_train_phase(K.dropout(ones, self.dropout_U), ones) for _ in range(4)]
            constants.append(B_U)
        else:
            constants.append([K.cast_to_floatx(1.) for _ in range(4)])

        if 0 < self.dropout_W < 1:
            input_shape = self.input_spec[0].shape
            input_dim = input_shape[-1]
            ones = K.ones_like(K.reshape(x[:, 0, 0], (-1, 1)))
            ones = K.concatenate([ones] * input_dim, 1)
            B_W = [K.in_train_phase(K.dropout(ones, self.dropout_W), ones) for _ in range(4)]
            constants.append(B_W)
        else:
            constants.append([K.cast_to_floatx(1.) for _ in range(4)])
        return constants

    def get_config(self):
        config = {"output_dim": self.output_dim,
                  "init": self.init.__name__,
                  "inner_init": self.inner_init.__name__,
                  "forget_bias_init": self.forget_bias_init.__name__,
                  "activation": self.activation.__name__,
                  "inner_activation": self.inner_activation.__name__,
                  "W_regularizer": self.W_regularizer.get_config() if self.W_regularizer else None,
                  "U_regularizer": self.U_regularizer.get_config() if self.U_regularizer else None,
                  "b_regularizer": self.b_regularizer.get_config() if self.b_regularizer else None,
                  "dropout_W": self.dropout_W,
                  "dropout_U": self.dropout_U}
        if self.batch_norm:
            config["momentum"] = self.momentum
        base_config = super(LSTM, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class BNLSTMCell(RNNCell):
    '''Batch normalized LSTM as described in arxiv.org/abs/1603.09025'''

    def __init__(self, num_units, training):
        self.num_units = num_units
        self.training = training

    @property
    def state_size(self):
        return (self.num_units, self.num_units)

    @property
    def output_size(self):
        return self.num_units

    def __call__(self, x, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            c, h = state

            x_size = x.get_shape().as_list()[1]
            W_xh = tf.get_variable('W_xh',
                                   [x_size, 4 * self.num_units],
                                   initializer=orthogonal_initializer())
            W_hh = tf.get_variable('W_hh',
                                   [self.num_units, 4 * self.num_units],
                                   initializer=bn_lstm_identity_initializer(0.95))
            bias = tf.get_variable('bias', [4 * self.num_units])

            xh = tf.matmul(x, W_xh)
            hh = tf.matmul(h, W_hh)

            bn_xh = batch_norm(xh, 'xh', self.training)
            bn_hh = batch_norm(hh, 'hh', self.training)

            hidden = bn_xh + bn_hh + bias

            i, j, f, o = tf.split(axis=1, num_or_size_splits=4, value=hidden)

            new_c = c * tf.sigmoid(f) + tf.sigmoid(i) * tf.tanh(j)
            bn_new_c = batch_norm(new_c, 'c', self.training)

            new_h = tf.tanh(bn_new_c) * tf.sigmoid(o)

            return new_h, (new_c, new_h)


def orthogonal(shape):
    flat_shape = (shape[0], np.prod(shape[1:]))
    a = np.random.normal(0.0, 1.0, flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    q = u if u.shape == flat_shape else v
    return q.reshape(shape)


def bn_lstm_identity_initializer(scale):
    def _initializer(shape, dtype=tf.float32, partition_info=None):
        '''Ugly cause LSTM params calculated in one matrix multiply'''
        size = shape[0]
        # gate (j) is identity
        t = np.zeros(shape)
        t[:, size:size * 2] = np.identity(size) * scale
        t[:, :size] = orthogonal([size, size])
        t[:, size * 2:size * 3] = orthogonal([size, size])
        t[:, size * 3:] = orthogonal([size, size])
        return tf.constant(t, dtype)

    return _initializer


def orthogonal_initializer():
    def _initializer(shape, dtype=tf.float32, partition_info=None):
        return tf.constant(orthogonal(shape), dtype)

    return _initializer


def batch_norm(x, name_scope, training, epsilon=1e-3, decay=0.99):
    '''Assume 2d [batch, values] tensor'''

    with tf.variable_scope(name_scope):
        size = x.get_shape().as_list()[1]

        scale = tf.get_variable('scale', shape=[size], initializer=tf.constant_initializer(0.1))
        offset = tf.get_variable('offset', shape=[size])

        pop_mean = tf.get_variable('pop_mean', shape=[size], initializer=tf.zeros_initializer(), trainable=False)
        pop_var = tf.get_variable('pop_var', shape=[size], initializer=tf.ones_initializer(), trainable=False)
        batch_mean, batch_var = tf.nn.moments(x, [0])

        train_mean_op = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))
        train_var_op = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay))

        def batch_statistics():
            with tf.control_dependencies([train_mean_op, train_var_op]):
                return tf.nn.batch_normalization(x, batch_mean, batch_var, offset, scale, epsilon)

        def population_statistics():
            return tf.nn.batch_normalization(x, pop_mean, pop_var, offset, scale, epsilon)

        return tf.cond(training, batch_statistics, population_statistics)
