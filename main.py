from pprint import pprint
import os

import tensorflow as tf

import read_data
import read_data_amazon
from models.n2n_DMN.dmn import n2nModel

flags = tf.app.flags

# File directories
flags.DEFINE_string("log_dir", "log", "Log directory [log]")
flags.DEFINE_string("data_dir", 'data/babi/en/', "Data folder directory [data/babi/en]")
flags.DEFINE_string("amazon_data_dir", 'data/amazon/', "Amazon data folder [data/amazon/]")
flags.DEFINE_string("save_dir", "save", "Save path [save]")

# Common training parameters
flags.DEFINE_integer("batch_size", 1, "Batch size during training and testing [32]")
flags.DEFINE_float("init_mean", 0, "Initial weight mean [0]")
flags.DEFINE_float("init_std", 0.1, "Initial weight std [0.1]")
flags.DEFINE_float("init_lr", 0.01, "Initial learning rate [0.01]")
flags.DEFINE_integer("num_epochs", 15, "Total number of epochs for training [100]")
flags.DEFINE_float("anneal_ratio", 0.5, "Annealing ratio [0.5]")
flags.DEFINE_integer("anneal_period", 25, "Number of epochs for every annealing [25]")

# Common options
flags.DEFINE_boolean("train", True, "Train? Test if False [False]")
flags.DEFINE_boolean("load", False, "Load from saved model? [False]")
flags.DEFINE_boolean("progress", True, "Show progress? [True]")
flags.DEFINE_boolean("gpu", False, 'Enable GPU? (Linux only) [False]')
flags.DEFINE_integer("val_period", 5, "Val period (for display purpose only) [5]")
flags.DEFINE_integer("save_period", 15, "Save period [10]")

# Debugging
flags.DEFINE_boolean("draft", False, "Draft? (quick build) [False]")

# Specific training parameters
flags.DEFINE_integer("memory_size", 100, "Memory size [50]")
flags.DEFINE_integer("hidden_size", 30, "Embedding dimension [20]")
flags.DEFINE_integer("num_layers", 3, "Number of memory layers (hops) [3]")
flags.DEFINE_boolean("linear_start", False, "Start training with linear model? [False]")
flags.DEFINE_float("ls_init_lr", 0.005, "Initial learning rate for linear start [0.005]")
flags.DEFINE_integer("ls_num_epochs", 20, "Linear start duration [20]")
flags.DEFINE_float("max_grad_norm", 40, "Max gradient norm; above this number is clipped [40]")
flags.DEFINE_boolean("position_encoding", True, "Position encoding enabled? 'True' or 'False' [True]")
flags.DEFINE_string("tying", 'adj', "Indicate tying method: 'adj' or 'rnn' [adj]")

# Specific options
flags.DEFINE_string("data_group",'babi',"Running model on: 'babi' or 'amazon' [babi]")
flags.DEFINE_integer("task", 1, "Task number [1]")
flags.DEFINE_float("val_ratio", 0.1, "Validation data ratio to training data [0.1]")

FLAGS = flags.FLAGS


def main(_):
    # create train and test data w/ batch_size and task #
    ''' Controls loading the data set and creating the training/testing formats '''
    if FLAGS.data_group == 'babi':
        (train_ds, test_ds), idx_to_word = read_data.read_babi(1, FLAGS.data_dir, FLAGS.task)
        train_ds, val_ds = read_data.split_val(train_ds, FLAGS.val_ratio)
        train_ds.name, val_ds.name, test_ds.name = 'train', 'val', 'test'
    else:
        train_ds, test_ds = read_data_amazon.read_amazon_split(FLAGS.batch_size)
        train_ds, val_ds = read_data_amazon.split_val_amazon(train_ds, FLAGS.val_ratio)
        train_ds.name, val_ds.name, test_ds.name = 'train', 'val', 'test'

    FLAGS.vocab_size = test_ds.vocab_size # get the size of the vocabulary
    ##TODO check if get_max_sizes needs to be adjusted to make this all work correctly
    FLAGS.max_sent_size, FLAGS.max_ques_size = read_data.get_max_sizes(train_ds, val_ds, test_ds)
    # FIXME : adhoc for now!``
    FLAGS.max_sent_size = max(FLAGS.max_sent_size, FLAGS.max_ques_size)
    FLAGS.train_num_batches = train_ds.num_batches
    FLAGS.val_num_batches = val_ds.num_batches
    FLAGS.test_num_batches = test_ds.num_batches
    if not os.path.exists(FLAGS.save_dir):
        os.mkdir(FLAGS.save_dir)

    if FLAGS.linear_start:
        FLAGS.num_epochs = FLAGS.ls_num_epochs
        FLAGS.init_lr = FLAGS.ls_init_lr

    if FLAGS.draft:
        FLAGS.num_layers = 1
        FLAGS.num_epochs = 1
        FLAGS.eval_period = 1
        FLAGS.ls_duration = 1
        FLAGS.train_num_batches = 1
        FLAGS.test_num_batches = 1
        FLAGS.save_period = 1

    # pprint(FLAGS.__flags)

    graph = tf.Graph()
    model = n2nModel(graph, FLAGS)
    with tf.Session(graph=graph) as sess:
        sess.run(tf.initialize_all_variables())
        if FLAGS.train:
            writer = tf.train.SummaryWriter(FLAGS.log_dir, sess.graph)
            if FLAGS.load:
                model.load(sess)
            model.train(sess, writer, train_ds, val_ds, idx_to_word)
        else:
            model.load(sess)
            model.eval(sess, test_ds, idx_to_word)


if __name__ == "__main__":
    tf.app.run()
