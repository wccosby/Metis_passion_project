"""
Implements the general structure of the dynamic memory network
"""

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import seq2seq, rnn_cell
from tensorflow.nn.rnn_cell import GRUCell


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

        """ build input module """
        print("==> Building input module")
        # reset gate
        self.W_input_reset = tf.Variable(tf.truncated_normal([self.dim, self.word_vector_size],stddev=0.2,-1,1),name="W_input_reset")
        self.U_input_reset = tf.Variable(tf.truncated_normal([self.dim, self.dim],stddev=0.2,-1,1),name="U_input_reset")
        self.b_input_reset = tf.Variable(tf.zeros([self.dim]), name="b_input_reset")

        # update gate
        self.W_input_update = tf.Variable(tf.truncated_normal([self.dim, self.word_vector_size],stddev=0.2,-1,1),name="W_input_update")
        self.U_input_update = tf.Variable(tf.truncated_normal([self.dim, self.dim],stddev=0.2,-1,1),name="U_input_update")
        self.b_input_update = tf.Variable(tf.zeros([self.dim]), name="b_input_update")

        # hidden state
        self.W_input_hidden = tf.Variable(tf.truncated_normal([self.dim, self.word_vector_size],stddev=0.2,-1,1),name="W_input_hidden")
        self.U_input_hidden = tf.Variable(tf.truncated_normal([self.dim, self.dim],stddev=0.2,-1,1),name="U_input_hidden")
        self.b_input_hidden = tf.Variable(tf.zeros([self.dim]), name="b_input_hidden")


        ##TODO a little unclear on what inp_c_history actually is
            ## --> records what memories have been recorded
        # returns a tensor shaped [batch_size, max_time, cell.output_size]
        input_c_history, _ = tf.nn.dynamic_rnn(
                                cell = self.input_gru_step, # neuron type/step
                                inputs = self.input_var, # data
                                dtype=tf.float32)

        self.input_c = input_c_history.take(self.input_mask_var, axis=0)

        self.q_q, _ = tf.nn.dynamic_rnn(
                                cell = self.input_gru_step, # neuron type/step
                                inputs = self.input_var, # data
                                dtype=tf.float32)

        # get the output of the last step of the GRU
        self.q_q = self.q_q[-1]

        """ Episodic memory module """
        print("==> Buidling components for episodic memory module")
        ## build episodic memory module
        # make parameters for memory module --> memory module also uses a GRU
        # memory reset gate
        self.W_memory_reset = tf.Variable(tf.truncated_normal([self.dim, self.dim],stddev=0.2,-1,1),name="W_memory_reset")
        self.U_memory_reset = tf.Variable(tf.truncated_normal([self.dim, self.dim],stddev=0.2,-1,1),name="U_memory_reset")
        self.b_memory_reset = tf.Variable(tf.zeros([self.dim]), name="b_memory_reset")

        # memory update gate
        self.W_memory_reset = tf.Variable(tf.truncated_normal([self.dim, self.dim],stddev=0.2,-1,1),name="W_memory_update")
        self.U_memory_reset = tf.Variable(tf.truncated_normal([self.dim, self.dim],stddev=0.2,-1,1),name="U_memory_update")
        self.b_memory_reset = tf.Variable(tf.zeros([self.dim]), name="b_memory_reset")

        # memory hidden
        self.W_memory_hidden = tf.Variable(tf.truncated_normal([self.dim, self.dim],stddev=0.2,-1,1),name="W_memory_hidden")
        self.U_memory_hidden = tf.Variable(tf.truncated_normal([self.dim, self.dim],stddev=0.2,-1,1),name="U_memory_hidden")
        self.b_memory_hidden = tf.Variable(tf.zeros([self.dim]), name="b_memory_hidden")

        # TODO figure out what this is
        ## attention mechanism, standard 2 layer neural network
        self.W_b = tf.Variable(tf.truncated_normal([self.dim, self.dim],sttdev=0.1))
        self.W_1 = tf.Variable(tf.truncated_normal([self.dim, 7*self.dim+2],sttdev=0.1))
        self.W_2 = tf.Variable(tf.truncated_normal([1, self.dim],stddev=0.1))
        self.b_1 = tf.Variable(tf.zeros([self.dim]))
        self.b_2 = tf.Variable(tf.zeros([1]))

        print("==> Building episodic memory module (fixed number of steps: %d)" % self.memory_hops)
        memory = [self.q_q.copy()] # add the question into the memory
        for iter in range(1, self.memory_hops + 1):
            current_episode = self.new_episode(memory[iter-1]) # make a new episode based on previous memory
        ## build answer module

        print "==> building episodic memory module (fixed number of steps: %d)" % self.memory_hops
        memory = [self.q_q.copy()]
        for iter in range(1, self.memory_hops + 1):
            current_episode = self.new_episode(memory[iter - 1])
            memory.append(self.GRU_update(memory[iter - 1], current_episode,
                                          self.W_mem_res_in, self.W_mem_res_hid, self.b_mem_res,
                                          self.W_mem_upd_in, self.W_mem_upd_hid, self.b_mem_upd,
                                          self.W_mem_hid_in, self.W_mem_hid_hid, self.b_mem_hid))

        last_mem = memory[-1]

        print "==> building answer module"
        self.W_a = nn_utils.normal_param(std=0.1, shape=(self.vocab_size, self.dim))

        if self.answer_module == 'feedforward':
            self.prediction = nn_utils.softmax(T.dot(self.W_a, last_mem))

        elif self.answer_module == 'recurrent':
            self.W_ans_res_in = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim + self.vocab_size))
            self.W_ans_res_hid = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim))
            self.b_ans_res = nn_utils.constant_param(value=0.0, shape=(self.dim,))

            self.W_ans_upd_in = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim + self.vocab_size))
            self.W_ans_upd_hid = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim))
            self.b_ans_upd = nn_utils.constant_param(value=0.0, shape=(self.dim,))

            self.W_ans_hid_in = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim + self.vocab_size))
            self.W_ans_hid_hid = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim))
            self.b_ans_hid = nn_utils.constant_param(value=0.0, shape=(self.dim,))

            def answer_step(prev_a, prev_y):
                a = self.GRU_update(prev_a, T.concatenate([prev_y, self.q_q]),
                                  self.W_ans_res_in, self.W_ans_res_hid, self.b_ans_res,
                                  self.W_ans_upd_in, self.W_ans_upd_hid, self.b_ans_upd,
                                  self.W_ans_hid_in, self.W_ans_hid_hid, self.b_ans_hid)

                y = nn_utils.softmax(T.dot(self.W_a, a))
                return [a, y]

            # TODO: add conditional ending
            dummy = theano.shared(np.zeros((self.vocab_size, ), dtype=floatX))
            results, updates = theano.scan(fn=answer_step,
                outputs_info=[last_mem, T.zeros_like(dummy)],
                n_steps=1)
            self.prediction = results[1][-1]

        else:
raise Exception("invalid answer_module")












    def GRU_update(self, prev_h, x, W_reset, U_reset, b_reset,
                    W_update, U_update, b_update,
                    W_hidden, U_hidden, b_hidden):
        """ Math operations for each gru step """
        z = tf.sigmoid(tf.matmul(W_update, x) + tf.matmul(U_update, prev_h) + b_update)
        r = tf.sigmoid(tf.matmul(W_reset, x) + tf.matmul(U_reset, prev_h) + b_update)
        _h = tf.tanh(tf.matmul(W_hidden, x) + tf.matmul(U_hidden, prev_h) + b_hidden)
        h_new = z * prev_h + (1-z) * _h
        return h_new


    def input_gru_step(self, x, prev_h):
        return self.GRU_update(prev_h, x, self.W_input_reset, self.U_input_reset, self.b_input_reset,
                                    self.W_input_update, self.U_input_update, self.b_input_update,
                                    self.W_input_hidden, self.U_input_hidden, self.b_input_hidden)

    def new_attention_step(self, ct, prev_g, mem, q_q):
        pass

    def new_episode_step(self, ct, g, prev_h):
        pass

    # TODO figure out episode steps a little/lot better
    def new_episode(self, mem):
        g,g_updates = tf.nn.dynamic_rnn(
                        cell = self.new_attention_step,
                        inputs = self.input_c,


        )
        inputs = self.input_var, # data
        dtype=tf.float32)




            g, g_updates = theano.scan(fn=self.new_attention_step,
            sequences=self.inp_c,
            non_sequences=[mem, self.q_q],
            outputs_info=T.zeros_like(self.inp_c[0][0]))

        if (self.normalize_attention):
            g = nn_utils.softmax(g)

        e, e_updates = theano.scan(fn=self.new_episode_step,
            sequences=[self.inp_c, g],
            outputs_info=T.zeros_like(self.inp_c[0]))

return e[-1]

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
