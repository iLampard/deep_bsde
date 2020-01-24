""" Load equation data and pre-defined terminal conditions """

import numpy as np
from scipy.stats import multivariate_normal as normal
import tensorflow as tf


class DataSet:
    def __init__(self, input_x, input_dw, batch_size):
        self.input_x = input_x
        self.input_dw = input_dw
        self.batch_size = batch_size
        self.max_num_batch = int(np.ceil(len(self.input_x) / self.batch_size))

    def get_batch_data(self):
        """ Return batch data - x and dw  """
        for i in range(self.max_num_batch):
            batch_data = []
            batch_x = self.input_x[i: i + self.batch_size]
            batch_dw = self.input_dw[i: i + self.batch_size]
            batch_data.append(batch_x)
            batch_data.append(batch_dw)
            yield batch_data

    @property
    def num_samples(self):
        """ Num of total samples """
        return len(self.input_x)


class BaseEquationLoader:
    """ Base class to load pre-defined equation """
    name = 'base_equation'

    def __init__(self):
        pass

    @property
    def dim(self):
        raise NotImplementedError

    @property
    def step_size(self):
        raise NotImplementedError

    def sample(self, num_sample):
        """ Overwrite this in derived class  """
        raise NotImplementedError

    def non_hmg_function(self, t, x, y, z):
        """ Non-homogenous term in BSDE => the drift term of y """
        raise NotImplementedError

    def terminal_condition(self, t, x):
        """ Termnal condition u(T, x) """
        raise NotImplementedError

    def process_data(self, num_sample):
        """ To overwrite in derived class  """
        raise NotImplementedError

    @staticmethod
    def get_loader_from_flags(total_time, num_steps, config):
        """ Find out correct derived class loader  """
        loader_cls = None
        equation_name = config.name
        for sub_loader_cls in BaseEquationLoader.__subclasses__():
            if sub_loader_cls.name == equation_name:
                loader_cls = sub_loader_cls

        if loader_cls is None:
            raise RuntimeError('Unknown equation name:' + equation_name)

        attributes = dict((name, getattr(config, name)) for name in dir(config)
                          if not name.startswith('__'))

        return loader_cls(total_time, num_steps, **attributes)

    @staticmethod
    def split_train_test(total_data, valid_start_ratio=0.8, test_start_ratio=0.9):
        """ Split the train, valid and test set """
        idx_train_end = int(valid_start_ratio * len(total_data))
        idx_valid_end = int(test_start_ratio * len(total_data))
        train_data = total_data[0: idx_train_end]
        valid_data = total_data[idx_train_end: idx_valid_end]
        test_data = total_data[idx_valid_end:]
        return train_data, valid_data, test_data

    def load_dataset(self, num_sample, batch_size):
        """ Return dataset object of train, valid and test set """
        self.train_data, self.valid_data, self.test_data = self.process_data(num_sample)
        train_dataset = DataSet(self.train_data[0], self.train_data[1], batch_size)
        valid_dataset = DataSet(self.valid_data[0], self.valid_data[1], batch_size)
        test_dataset = DataSet(self.test_data[0], self.test_data[1], batch_size)

        return train_dataset, valid_dataset, test_dataset


class BSEquationLoader(BaseEquationLoader):
    name = 'BlackScholes'

    def __init__(self,
                 total_time,
                 num_steps,
                 **kwargs):
        super(BSEquationLoader, self).__init__()
        self.total_time = total_time
        self.num_steps = num_steps
        self.delta_time = total_time / num_steps
        self.equation_dim = kwargs.get('equation_dim')
        self.sigma = kwargs.get('sigma')
        self.mu = kwargs.get('mu')
        self.spot = kwargs.get('spot')
        self.strike = kwargs.get('strike')
        self.x_init = np.ones(self.equation_dim) * self.spot
        self.sqrt_delta_time = np.sqrt(self.delta_time)
        self.init_y_range = kwargs.get('init_y_range', [0, 10])

    @property
    def dim(self):
        return self.equation_dim

    @property
    def step_size(self):
        return self.delta_time

    def non_hmg_function(self, t, x, y, z):
        return -self.mu * y

    def terminal_condition(self, t, x):
        return tf.maximum(x - self.strike, 0.0)

    def sample(self, num_sample):
        """ Sample dw and x """
        # (num_sample, num_steps, equation_dim)
        sample_dw = normal.rvs(size=[num_sample,
                                     self.num_steps,
                                     self.equation_dim]) * self.sqrt_delta_time

        sample_dw = sample_dw.reshape((num_sample, self.num_steps, self.equation_dim))

        # (num_sample, num_steps, equation_dim)
        sample_x = np.zeros([num_sample, self.num_steps + 1, self.equation_dim])
        sample_x[:, 0, :] = np.ones([num_sample, self.equation_dim]) * self.x_init
        factor = np.exp((self.mu - (self.sigma ** 2) / 2) * self.delta_time)

        for i in range(self.num_steps):
            sample_x[:, i + 1, :] = factor * np.exp(self.sigma * sample_dw[:, i, :]) \
                                    * sample_x[:, i, :]

        return sample_x, sample_dw

    def process_data(self, num_sample):
        """ Generate train, valid and test set  """
        sample_x, sample_dw = self.sample(num_sample)

        train_x, valid_x, test_x = self.split_train_test(sample_x)
        train_dw, valid_dw, test_dw = self.split_train_test(sample_dw)

        return (train_x, train_dw), (valid_x, valid_dw), (test_x, valid_dw)
