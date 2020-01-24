""" A script to run the model """

import os
import sys

CUR_PATH = os.path.abspath(os.path.dirname(__file__))
ROOT_PATH = os.path.split(CUR_PATH)[0]
sys.path.append(ROOT_PATH)

from absl import app
from absl import flags
from deep_bsde.model_runner import ModelRunner
from deep_bsde.model.bsde import DeepBSDE
from deep_bsde.config_loader import BaseConfigLoader
from deep_bsde.equation_loader import BaseEquationLoader

FLAGS = flags.FLAGS

# Data input params
flags.DEFINE_string('problem', 'BlackScholes', 'Name of PDE to solve')

# Model runner params
flags.DEFINE_bool('write_summary', False, 'Whether to write summary of epoch in training using Tensorboard')
flags.DEFINE_integer('max_epoch', 100, 'Max epoch number of training')
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate')
flags.DEFINE_integer('batch_size', 128, 'Batch size of data fed into model')
flags.DEFINE_integer('num_sample', 10000, 'Total samples as the input of the neural model')

# Model params
flags.DEFINE_enum('network_type',
                  'stepwise_net',
                  ['stepwise_net', 'merged'],
                  'Network type for the evolution of the dynamics')
flags.DEFINE_float('total_time', 1.0, 'Length of total time')
flags.DEFINE_bool('apply_bn', False, 'Whether to apply batch norm in sublayers')
flags.DEFINE_integer('hidden_dim', 32, 'Dimention of hidden layers')
flags.DEFINE_integer('num_steps', 50, 'Num of time steps for input x data')
flags.DEFINE_string('save_dir', 'logs', 'Root path to save logs and models')


def main(argv):
    config_loader = BaseConfigLoader.get_loader_from_flags(FLAGS.problem)

    equation_loader = BaseEquationLoader.get_loader_from_flags(FLAGS.total_time,
                                                               FLAGS.num_steps,
                                                               config_loader)

    train_set, valid_set, test_set = equation_loader.load_dataset(FLAGS.num_sample,
                                                                  FLAGS.batch_size)

    model = DeepBSDE(hidden_dim=FLAGS.hidden_dim,
                     bsde=equation_loader,
                     network_type=FLAGS.network_type,
                     apply_bn=FLAGS.apply_bn)


    # Model
    model_runner = ModelRunner(model, FLAGS)

    model_runner.train(train_set, valid_set, test_set, FLAGS.max_epoch)

    model_runner.evaluate(test_set)

    return


if __name__ == '__main__':
    app.run(main)
