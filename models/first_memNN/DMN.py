"""
Implements the general structure of the dynamic memory network
"""

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import seq2seq, rnn_cell


class DMN_simple:

    def __init__(self, babi_train_raw, babi_test_raw, word2vec, word_vector_dim,
                dim, mode, answer_module, input_mask_mode, memory_hops, L2,
                normalize_attention, **kwargs):
        """initialize the variables and build the network"""
        ## initialize stuff
        ## Need
        # all the things from above
        self.word2vec = utils.load_glove(word_vector_dim)
        self.word_vector_dim = word_vector_dim
        self.dim = dim
        self.mode = mode
        self.answer_module = answer_module
        self.input_mask_mode = input_mask_mode
        self.memory_hops = memory_hops
        self.L2 = L2
        self.normalize_attention = normalize_attention

        # vocab dict
        # ivocab dict (indexes)
        self.vocab = {}
        self.ivocab = {}

        # training input, question, answer, and input mask from process input function
        # testing for the same (passing in test set instead of training set)
        self.train_input, self.train_question, self.train_answer, self.train_input_mask = self._process_input(babi_train_raw)
        self.test_input, self.test_question, self.test_answer, self.test_input_mask = self._process_input(babi_test_raw)
        # vocabulary size
        self.vocab_size = len(self.vocab)

        # tensorflow variable initialization
        ## placeholders of inputs use the sizes of the train input, question, answer, and mask

        ## can use feed_dicts to put data into these things later on
        self.input_var = tf.placeholder(tf.float32, None, name='input_var') #TODO not sure about size of vector here
        self.question_var = tf.placeholder(tf.float32, None, name='question_var')
        self.answer_var = tf.placeholder(tf.float32,None, name='answer_var')
        self.input_mask = tf.placeholder(tf.int32,None, name='input_mask')

        ## build input module




    self.W_inp_res_in = nn_utils.normal_param(std=0.1, shape=(self.dim, self.word_vector_size))
    self.W_inp_res_hid = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim))
    self.b_inp_res = nn_utils.constant_param(value=0.0, shape=(self.dim,))

        ## build episodic memory module

        ## build answer module


    def GRU_update(self, h, x, W_reset, U_reset, b_reset,
                    W_update, U_update, b_update,
                    W_hidden, U_hidden, b_hidden):
        pass

    def input_gru_step(self, x, prev_h):
        pass

    def new_attention_step(self, ct, prev_g, mem, q_q):
        pass

    def new_episode_step(self, ct, g, prev_h):
        pass

    def new_episode(self, mem):
        pass

    def save_params(self, file_name, epoch, **kwargs):
        with open(file_name, 'w') as save_file:
            pickle.dump(
                obj = {
                    'params' : [x.get_value() for x in self.params],
                    'epoch' : epoch,
                    'gradient_value': (kwargs['gradient_value'] if 'gradient_value' in kwargs else 0)
                },
                file = save_file,
                protocol = -1
            )

    def load_state(self, file_name):
        print "==> loading state %s" % file_name
        with open(file_name, 'r') as load_file:
            dict = pickle.load(load_file)
            loaded_params = dict['params']
            for (x, y) in zip(self.params, loaded_params):
                x.set_value(y)

    def _process_input(self, data_raw):
        questions = []
        inputs = []
        answers = []
        input_masks = []
        for x in data_raw:
            # get the "story"
            inp = x["C"].lower().split(' ')
            # add each word in the story to the word list for the input
            inp = [w for w in inp if len(w) > 0]
            # get the question
            q = x["Q"].lower().split(' ')
            # add each word in the story to word list for the question
            q = [w for w in q if len(w) > 0]

            # make the vector for the input
            input_vector = [utils.process_word(word = w,
                                        word2vec = self.word2vec,
                                        vocab = self.vocab,
                                        ivocab = self.ivocab,
                                        word_vector_size = self.word_vector_dim,
                                        to_return = "word2vec") for w in inp]

            inputs.append(np.vstack(inp_vector).astype(float))

            q_vector = [utils.process_word(word = w,
                                        word2vec = self.word2vec,
                                        vocab = self.vocab,
                                        ivocab = self.ivocab,
                                        word_vector_size = self.word_vector_dim,
                                        to_return = "word2vec") for w in q]

            questions.append(np.vstack(q_vector).astype(float))

            answers.append(utils.process_word(word = x["A"],
                                            word2vec = self.word2vec,
                                            vocab = self.vocab,
                                            ivocab = self.ivocab,
                                            word_vector_size = self.word_vector_dim,
                                            to_return = "index"))
            # NOTE: here we assume the answer is one word!
            if self.input_mask_mode == 'word':
                input_masks.append(np.array([index for index, w in enumerate(inp)], dtype=np.int32))
            elif self.input_mask_mode == 'sentence':
                input_masks.append(np.array([index for index, w in enumerate(inp) if w == '.'], dtype=np.int32))
            else:
                raise Exception("invalid input_mask_mode")

return inputs, questions, answers, input_masks

    def get_batches_per_epoch(self,mode):
        pass

    def shuffle_train_set(self):
        pass

    def step(self, batch_index, mode):
        pass
