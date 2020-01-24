""" Main model definition  """

import tensorflow as tf
from tensorflow.keras import layers


def batch_norm(inputs,
               scope,
               offset=0,
               scale=0.1,
               decay=0.99,
               variance_epsilon=1e-5,
               is_training=True):
    with tf.variable_scope(scope):
        input_dim = inputs.get_shape().as_list()[-1]
        pop_mean = tf.get_variable(name='pop_mean',
                                   shape=[input_dim],
                                   initializer=tf.zeros_initializer())

        pop_var = tf.get_variable(name='pop_var',
                                  shape=[input_dim],
                                  initializer=tf.ones_initializer())
        batch_mean, batch_var = tf.nn.moments(inputs, [0])

        def batch_statistics():
            pop_mean_new = pop_mean * decay + batch_mean * (1 - decay)
            pop_var_new = pop_var * decay + batch_var * (1 - decay)
            with tf.control_dependencies([pop_mean.assign(pop_mean_new),
                                          pop_var.assign(pop_var_new)]):
                return tf.nn.batch_normalization(inputs,
                                                 batch_mean,
                                                 batch_var,
                                                 offset,
                                                 scale,
                                                 variance_epsilon)

        def population_statistics():
            return tf.nn.batch_normalization(inputs,
                                             pop_mean,
                                             pop_var,
                                             offset,
                                             scale,
                                             variance_epsilon)

        return tf.cond(is_training, batch_statistics, population_statistics)


class SubLayerAtEachStep(layers.Layer):
    def __init__(self, hidden_dim, bsde_dim, is_training, apply_bn):
        super(SubLayerAtEachStep, self).__init__()
        self.hidden_dim = hidden_dim
        self.bsde_dim = bsde_dim
        self.is_training = is_training
        self.apply_bn = apply_bn
        self.layer_input = layers.Dense(self.hidden_dim)
        self.layer_fc = layers.Dense(self.hidden_dim,
                                     activation=tf.nn.relu)
        self.layer_fc_2 = layers.Dense(self.hidden_dim,
                                       activation=tf.nn.relu)
        self.layer_output = layers.Dense(self.bsde_dim)

    def call(self, inputs):
        """ Pass through a few dense layers """
        output = self.layer_input(inputs)
        if self.apply_bn:
            output = batch_norm(output, 'bn_1', is_training=self.is_training)
        output = self.layer_fc(output)
        if self.apply_bn:
            output = batch_norm(output, 'bn_2', is_training=self.is_training)
        output = self.layer_fc_2(output)
        if self.apply_bn:
            output = batch_norm(output, 'bn_3', is_training=self.is_training)
        output = self.layer_output(output)
        if self.apply_bn:
            output = batch_norm(output, 'bn_4', is_training=self.is_training)

        return output


class DeepBSDE:
    def __init__(self, hidden_dim, bsde, network_type, apply_bn):
        self.hidden_dim = hidden_dim
        self.bsde = bsde
        self.input_x_dim = self.bsde.equation_dim
        self.network_type = network_type
        self.apply_bn = apply_bn

    def build(self):
        """ Build up the network  """
        with tf.variable_scope('DeepBSDE'):
            # (batch_size, num_steps, num_series)
            self.input_x = tf.placeholder(tf.float32, shape=[None, None, self.input_x_dim])
            self.input_dw = tf.placeholder(tf.float32, shape=[None, None, self.input_x_dim])

            self.lr = tf.placeholder(tf.float32)
            self.is_training = tf.placeholder(tf.bool)

            # (batch_size, 1)
            self.init_y = tf.get_variable(name='init_y',
                                          shape=[1],
                                          initializer=tf.random_uniform_initializer(maxval=self.bsde.init_y_range[1],
                                                                                    minval=self.bsde.init_y_range[0]))

            # (batch_size, bsde_dim)
            self.init_z = tf.get_variable(name='init_z',
                                          shape=[1, self.bsde.dim],
                                          initializer=tf.random_uniform_initializer(maxval=0.1,
                                                                                    minval=-0.1))

            self.pred, self.loss = self.forward(self.input_x, self.input_dw)

            self.opt_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

            return

    def init_state(self, input_x, input_dw):
        # (batch_size, bsde_dim)
        init_x = input_x[:, 0, :]

        # (batch_size, 1)
        init_y = tf.ones([self.batch_size, 1]) * self.init_y

        # (batch_size, bsde_dim)
        init_z = tf.matmul(tf.ones([self.batch_size, 1]), self.init_z)

        # (batch_size, bsde_dim)
        init_w = input_dw[:, 0, :]

        return init_x, init_y, init_z, init_w

    def update_y(self, x, y, z, idx, dw):
        return y - self.bsde.non_hmg_function(idx * self.bsde.dim, x, y, z) * \
               self.bsde.step_size + tf.reduce_sum(z * dw, axis=-1, keepdims=True)

    def one_step_subnet_stepwise(self, prev_tuple, current_input):
        # x, y, z, at previous step
        x, y, z, dw, idx = prev_tuple

        # Current x and dw
        input_x, input_dw = tf.unstack(current_input, axis=0)

        # Update y
        y = self.update_y(x, y, z, idx, dw)

        # Update z
        z = SubLayerAtEachStep(self.hidden_dim, self.bsde.dim, self.is_training, self.apply_bn)(x)

        # Update x
        x = input_x

        # Update dw
        dw = input_dw

        return (x, y, z, dw, idx + 1)

    def all_steps_subnet_stepwise(self, input_x, input_dw):
        # (batch_size, bsde_dim) for x, z, dw
        # (batch_size, 1) for y
        init_x, init_y, init_z, init_dw = self.init_state(input_x, input_dw)

        # (num_steps, batch_size, bsde_dim)
        input_x_ = tf.transpose(input_x[:, 1:, :], perm=[1, 0, 2])

        # (num_steps - 1, batch_size, bsde_dim)
        input_dw_ = tf.transpose(input_dw[:, 1:, :], perm=[1, 0, 2])

        # Force dw tensor to have num_steps length while its last step will not be used in calculation
        # (num_steps, batch_size, bsde_dim)
        input_dw_ = tf.concat([input_dw_, input_dw_[-1:, :, :]], axis=0)

        # (num_steps, 2, batch_size, bsde_dim)
        stacked_input = tf.stack([input_x_, input_dw_], axis=1)

        # (num_steps, batch_size, bsde_dim)
        all_x, all_y, all_z, all_dw, all_idx = tf.scan(self.one_step_subnet_stepwise,
                                                       elems=stacked_input,
                                                       initializer=(init_x,
                                                                    init_y,
                                                                    init_z,
                                                                    init_dw,
                                                                    0.0))
        # (batch_size, 1)
        return all_y[-1]

    def one_step_subnet_merged(self, prev_tuple, current_input):
        # x, y, z, at previous step
        x, y, z, dw, idx = prev_tuple

        # Current x and dw
        input_x, input_dw = tf.unstack(current_input, axis=0)

        # Update y
        y = self.update_y(x, y, z, idx, dw)

        # Update z
        z = self.layer_merged(x)

        # Update x
        x = input_x

        # Update dw
        dw = input_dw

        return (x, y, z, dw, idx + 1)

    def all_steps_subnet_merged(self, input_x, input_dw):

        self.layer_merged = SubLayerAtEachStep(self.hidden_dim, self.bsde.dim, self.is_training, self.apply_bn)

        # (batch_size, bsde_dim) for x, z, dw
        # (batch_size, 1) for y
        init_x, init_y, init_z, init_dw = self.init_state(input_x, input_dw)

        # (num_steps, batch_size, bsde_dim)
        input_x_ = tf.transpose(input_x[:, 1:, :], perm=[1, 0, 2])

        # (num_steps - 1, batch_size, bsde_dim)
        input_dw_ = tf.transpose(input_dw[:, 1:, :], perm=[1, 0, 2])

        # Force dw tensor to have num_steps length while its last step will not be used in calculation
        # (num_steps, batch_size, bsde_dim)
        input_dw_ = tf.concat([input_dw_, input_dw_[-1:, :, :]], axis=0)

        # (num_steps, 2, batch_size, bsde_dim)
        stacked_input = tf.stack([input_x_, input_dw_], axis=1)

        # (num_steps, batch_size, bsde_dim)
        all_x, all_y, all_z, all_dw, all_idx = tf.scan(self.one_step_subnet_merged,
                                                       elems=stacked_input,
                                                       initializer=(init_x,
                                                                    init_y,
                                                                    init_z,
                                                                    init_dw,
                                                                    0.0))
        # (batch_size, 1)
        return all_y[-1]

    def forward(self, input_x, input_dw):
        """ Forward through time axis """

        self.batch_size = tf.shape(input_x)[0]
        self.num_steps = input_x.get_shape().as_list()[1]

        # (batch_size, 1)
        if self.network_type == 'stepwise_net':
            pred = self.all_steps_subnet_stepwise(input_x, input_dw)
        elif self.network_type == 'merged':
            pred = self.all_steps_subnet_merged(input_x, input_dw)

        # (batch_size, 1)
        terminal_time = self.bsde.step_size * self.bsde.num_steps
        label = self.bsde.terminal_condition(terminal_time,
                                             input_x[:, -1, :])

        loss = tf.reduce_mean(tf.square(pred - label))

        return pred, loss

    def train(self, sess, batch_data, lr):
        """ Define the train process """
        batch_x, batch_dw = batch_data
        feed_dict = {self.input_x: batch_x,
                     self.input_dw: batch_dw,
                     self.lr: lr,
                     self.is_training: True}

        _, loss, init_y = sess.run([self.opt_op, self.loss, self.init_y],
                                   feed_dict=feed_dict)

        return loss, init_y

    def predict(self, sess, batch_data):
        """ Define the prediction process """
        batch_x, batch_dw = batch_data
        feed_dict = {self.input_x: batch_x,
                     self.input_dw: batch_dw,
                     self.is_training: False}

        loss, init_y = sess.run([self.loss, self.init_y], feed_dict=feed_dict)

        return loss, init_y
