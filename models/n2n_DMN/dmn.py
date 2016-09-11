import tensorflow as tf
import numpy as np

from models.base_model import BaseModel

""" Just as a convenient way to pass around things i want to have more attributes but not their own class """
class Container(object):
    pass

class MemoryLayer(object):
    # phs == placeholders
    def __init__(self, params, prev_layer, phs, constants, tensors):
        self.params = params # getting the parameters specified from main.py
        # initialize basic sizes and dimensions of what will become the network
        batch_size = params.batch_size # batch size
        memory_size = params.memory_size # memory size TODO not clear of diff between this and d
        sent_size = params.max_sent_size # this is for sentence padding
        vocab_size = params.vocab_size # this V in the paper
        hidden_size = params.hidden_size # this is d in the paper (dim of memories)

        linear_start = params.linear_start # controls if softmaxes are removed initially vs having softmaxes from the beginning

        x_batch, x_mask_aug_batch, m_mask_batch = phs.x_batch, phs.x_mask_aug_batch, phs.m_mask_batch
        l_aug_aug = constants.l_aug_aug # positional encoding of input

        B, first_u_batch = tensors.B, tensors.first_u_batch # B is query embedding matrix, first_u_batch is first query


        """
        2 types of tying in the model:
            1) Adjacent:
                output embedding for one layer is input embedding for the one above (next)
                    A_(k+1) = C_k where k is the hop number
                    also constrain answer prediction matrix to be same as final output embedding --> W_T = C_k
                    constrain question embedding (B) to match input embedding of first layer: B = A_1
            2) Layer-wise (RNN-like):
                input and output embeddings are same across different layers: A_1 = A_2 = ... = A_K and C_1 = ... = C_K
                add linear mapping (H) to update query vector (u) between hops --> u_(k+1) = H * u_k + o_k
                    this mapping is learnt along with the rest of the parameters
        """
        #TODO here in the .get_variable functons pass in an "initializer=word_embedding_matrix"
        #NOTE the TA, TB, TC matrices are for the temporal encoding (i dont need to initialize them in any way)

        if not prev_layer: # if this is the first layer then we need to define the A and C embedding matrices
            if params.tying == 'adj':
                A = tf.identity(B, name='A')
            else:
                A = tf.get_variable('A',dtype='float', shape=[vocab_size,hidden_size])

            TA = tf.get_variable('TA',dtype='float',shape=[memory_size,hidden_size])
            C = tf.get_variable('C', dtype='float', shape=[vocab_size, hidden_size])
            TC = tf.get_variable('TC', dtype='float', shape=[memory_size, hidden_size])
        else:
            if params.tying == 'adj':
                A = tf.identity(prev_layer.C, name='A')
                TA = tf.identity(prev_layer.TC, name='TA')
                C = tf.get_variable('C', dtype='float', shape=[vocab_size, hidden_size])
                TC = tf.get_variable('TC', dtype='float', shape=[memory_size, hidden_size])
            elif params.tying == 'rnn':
                A = tf.identity(prev_layer.A, name='A')
                TA = tf.identity(prev_layer.TA, name='TA')
                C = tf.identity(prev_layer.C, name='C')
                TC = tf.identity(prev_layer.TC, name='TC')
            else:
                raise Exception('Unknown tying method: %s' % params.tying)

        if not prev_layer:
            u_batch = tf.identity(tensors.first_u_batch, name='u')
        else:
            u_batch = tf.add(prev_layer.u_batch, prev_layer.o_batch, name='u')

        with tf.name_scope('m'):
            # TODO adjust the embeddings here for pretrained word vectors
            Ax_batch = tf.nn.embedding_lookup(A, x_batch)  # [N, M, J, d]
            if params.position_encoding:
                Ax_batch *= l_aug_aug  # position encoding
            Ax_batch *= x_mask_aug_batch  # masking
            m_batch = tf.reduce_sum(Ax_batch, 2)  # [N, M, d]
            m_batch = tf.add(tf.expand_dims(TA, 0), m_batch, name='m')  # temporal encoding

        with tf.name_scope('c'):
            # TODO adjust embeddings here too for pretrained word vectors
            Cx_batch = tf.nn.embedding_lookup(C, x_batch)  # [N, M, J, d]
            if params.position_encoding:
                Cx_batch *= l_aug_aug  # position encoding
            Cx_batch *= x_mask_aug_batch
            c_batch = tf.reduce_sum(Cx_batch, 2)
            c_batch = tf.add(tf.expand_dims(TC, 0), c_batch, name='c')  # temporal encoding

        with tf.name_scope('p'):
            u_batch_aug = tf.expand_dims(u_batch, -1)  # [N, d, 1]
            um_batch = tf.squeeze(tf.batch_matmul(m_batch, u_batch_aug), [2])  # [N, M]
            if linear_start:
                p_batch = tf.mul(um_batch, m_mask_batch, name='p')
            else:
                p_batch = self._softmax_with_mask(um_batch, m_mask_batch)

        with tf.name_scope('o'):
            o_batch = tf.reduce_sum(c_batch * tf.expand_dims(p_batch, -1), 1)  # [N, d]

        self.A, self.TA, self.C, self.TC = A, TA, C, TC
        self.u_batch, self.o_batch = u_batch, o_batch

    def _softmax_with_mask(self, um_batch, m_mask_batch):
        exp_um_batch = tf.exp(um_batch)  # [N, M]
        masked_batch = exp_um_batch * m_mask_batch  # [N, M]
        sum_2d_batch = tf.expand_dims(tf.reduce_sum(masked_batch, 1), -1)  # [N, 1]
        p_batch = tf.div(masked_batch, sum_2d_batch, name='p')  # [N, M]
        # print "p_batch: ", p_batch
        return p_batch


class n2nModel(BaseModel):
    def _build_tower(self):
        params = self.params
        linear_start = params.linear_start # controls if softmaxes are removed initially vs having softmaxes from the beginning

        batch_size = params.batch_size # batch size (N in code)
        memory_size = params.memory_size # memory size TODO not clear of diff between this and d (M in code)
        sent_size = params.max_sent_size # this is for sentence padding (J in code)
        vocab_size = params.vocab_size # this V in the paper (V in code)
        hidden_size = params.hidden_size # this is d in the paper (dim of memories) (d in code)

        summaries = [] # list for summarized answers

        # initialize self
        # make the place holders for inputs
        with tf.name_scope('placeholders'):
            # define story/fact inputs
            with tf.name_scope('X'):
                x_batch = tf.placeholder('int32',shape=[batch_size, memory_size, sent_size], name='X')
                x_mask_batch = tf.placeholder('float',shape=[batch_size,memory_size, sent_size], name='X_mask')
                # need to change the dimensions so that multiplication with embedding matrix works --> adjustment because of batch sizing
                x_mask_aug_batch = tf.expand_dims(x_mask_batch, -1, 'x_mask_aug') # makes this shape=[batch_size, memory_size, sent_size, 1]
                m_mask_batch = tf.placeholder('float', shape=[batch_size, memory_size]) # define what will be the memory

            # define question inputs (shape=[N,J])
            with tf.name_scope('q'):
                q_batch = tf.placeholder('int32',shape=[batch_size, sent_size],name='q')
                q_mask_batch = tf.placeholder('float',shape=[batch_size, sent_size], name='q_mask')
                # need to change the dimensions so that multiplication with embedding matrix works --> adjustment because of batch sizing
                q_mask_aug_batch = tf.expand_dims(q_mask_batch, -1, 'q_mask_aug') # shape=[batch_size,sent_size,1] --> for multiplication with B


            # define the target (y_batch, shape=[N], int32)
            y_batch = tf.placeholder('int32', shape=[batch_size],name='y')
            # print "y_batch: ",y_batch

            # define learning rate
            learning_rate = tf.placeholder('float',name='learning_rate')

        ## define constants (l, l_aug, l_aug_aug)
        with tf.name_scope('constants'):
            l = self._get_l()
            l_aug = tf.expand_dims(l,0,name='l_aug')
            l_aug_aug = tf.expand_dims(l,0,name='l_aug_aug')

        # define answer layer...its a neural network embedding_lookup, shape[V]
        with tf.name_scope('answer'):
            ''' I think this compares the answer vector? '''
            # TODO what do we do about this
            a_batch = tf.nn.embedding_lookup(tf.diag(tf.ones(shape=[vocab_size])),y_batch,name='answer') # [batch_size, hidden_size]
            # print "a_batch: ", a_batch
        ## define what happens to the question embedding initially
        with tf.name_scope('first_u'):
            B = tf.get_variable('B',dtype='float',shape=[vocab_size, hidden_size])
            ## embedding_lookup --> looks up ids in a list of embedding tensors
            # here B is interpreted as a partition of of a larger embedding tensor (which is B)
            # q_batch here contains the ids to be looked up in B
            # returns tensor of the same type as tensors in params (which is B), has shape (shape(q_batch) + shape(B[1:]))
            # TODO i think the returned tensor for this should be [batch_size, sent_size, hidden_size] NOT vocab_size
            Bq_batch = tf.nn.embedding_lookup(B, q_batch) # [vocab_size, sent_size, hidden_size]
            # print("Shape of B --> ", tf.get_shape(Bq_batch))
            # TODO alter Bq_batch here for positonal encoding
            if params.position_encoding:
                Bq_batch *= l_aug
            Bq_batch *= q_mask_aug_batch #TODO wat is this --> shape=[batch_size,sent_size, hidden_size] *= [batch_size, sent_size, 1]
            first_u_batch = tf.reduce_sum(Bq_batch,1,name='first_u') # [batch_size, hidden_size]

        placeholders, constants, tensors = Container(), Container(), Container()
        placeholders.x_batch, placeholders.x_mask_batch, placeholders.x_mask_aug_batch, placeholders.m_mask_batch = x_batch, x_mask_batch, x_mask_aug_batch, m_mask_batch
        constants.l_aug_aug = l_aug_aug
        tensors.B, tensors.first_u_batch = B, first_u_batch

        memory_layers = []
        current_layer = None
        """ Construct the memory layers """
        for layer_num in range(params.num_layers):
            with tf.variable_scope('layer_%d' % layer_num):
                memory_layer = MemoryLayer(params, current_layer, placeholders, constants, tensors)
                memory_layers.append(memory_layer)
                current_layer = memory_layer

        """ get the last question vector """
        with tf.variable_scope('last_u'):
            if params.tying == 'rnn':
                H = tf.get_variable('H',shape=[hidden_size, hidden_size])
                last_u_batch = tf.add(tf.matmul(current_layer.u_batch,H), current_layer.o_batch, name='last_u')
            else:
                last_u_batch = tf.add(current_layer.u_batch, current_layer.o_batch, name='last_u')

        """ answer module """ ## TODO i think...
        with tf.variable_scope('ap'):
            if params.tying == 'adj':
                W = tf.transpose(current_layer.C, name='W')
            elif params.tying == 'rnn':
                W = tf.get_variable('W',dtype='float', shape=[hidden_size, vocab_size])
            else:
                raise Exception("Unsupported tying method: %s" % params.tying)
            logit_batch = tf.matmul(last_u_batch, W, name='logit')
            ap_batch = tf.nn.softmax(logit_batch, name='ap')
            # print ap_batch.eval(self._get_feed_dict)
            #TODO UGHAUJSDFJAWJEFAJWEFJWAJF FUUUUUUUUCCCCCKKKKKKKKK
            # print "ap_batch: ", ap_batch.eval()

        """ Define loss for the networks """
        with tf.name_scope("loss") as loss_scope:
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logit_batch, a_batch, name='cross_entropy')
            avg_cross_entropy = tf.reduce_mean(cross_entropy, 0, name="avg_cross_entropy")
            tf.add_to_collection('losses',avg_cross_entropy)
            total_loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
            losses = tf.get_collection('losses', loss_scope)

        with tf.name_scope('acc'):
            predicted = tf.argmax(ap_batch, 1)
            actual = tf.argmax(a_batch, 1)
            correct_vec = tf.equal(tf.argmax(ap_batch, 1), tf.argmax(a_batch, 1))
            num_corrects = tf.reduce_sum(tf.cast(correct_vec, 'float'), name='num_corrects')
            acc = tf.reduce_mean(tf.cast(correct_vec, 'float'), name='acc')

        with tf.name_scope('opt'):
            opt = tf.train.GradientDescentOptimizer(learning_rate)
            # FIXME : This must muse cross_entropy for some reason!
            grads_and_vars = opt.compute_gradients(cross_entropy)
            clipped_grads_and_vars = [(tf.clip_by_norm(grad, params.max_grad_norm), var) for grad, var in grads_and_vars]
            opt_op = opt.apply_gradients(clipped_grads_and_vars, global_step=self.global_step)

        # placeholders
        self.x = x_batch
        self.x_mask = x_mask_batch
        self.m_mask = m_mask_batch
        self.q = q_batch
        self.q_mask = q_mask_batch
        self.y = y_batch
        self.learning_rate = learning_rate

        # tensors
        self.total_loss = total_loss
        self.correct_vec = correct_vec
        self.predicted = predicted
        self.num_corrects = num_corrects
        self.acc = acc
        self.opt_op = opt_op
        self.actual = actual

        # summaries --> for tensorboard
        summaries.append(tf.scalar_summary("%s (raw)" % total_loss.op.name, total_loss))
        self.merged_summary = tf.merge_summary(summaries)

    ''' for positional encoding '''
    def _get_l(self):
        J, d = self.params.max_sent_size, self.params.hidden_size
        def f(JJ, jj, dd, kk):
            return (1-float(jj)/JJ) - (float(kk)/dd)*(1-2.0*jj/JJ)
        def g(jj):
            return [f(J, jj, d, k) for k in range(d)]
        l = [g(j) for j in range(J)]
        l_tensor = tf.constant(l, shape=[J, d], name='l')
        return l_tensor

    def _softmax_with_mask(self, um_batch, m_mask_batch):
        exp_um_batch = tf.exp(um_batch)  # [N, M]
        masked_batch = exp_um_batch * m_mask_batch  # [N, M]
        sum_2d_batch = tf.expand_dims(tf.reduce_sum(masked_batch, 1), -1)  # [N, 1]
        p_batch = tf.div(masked_batch, sum_2d_batch, name='p')  # [N, M]
        return p_batch

    """
    loads the placeholders with data
    """
    def _get_feed_dict(self,batch):
        sent_batch, ques_batch = batch[:2]
        if len(batch) > 2:
            label_batch = batch[2]
        else:
            label_batch = np.zeros([len(sent_batch)])
        x_batch, x_mask_batch, m_mask_batch = self._prepro_sent_batch(sent_batch)
        q_batch, q_mask_batch = self._prepro_ques_batch(ques_batch)
        y_batch = self._prepro_label_batch(label_batch)
        feed_dict = {self.x: x_batch, self.x_mask: x_mask_batch, self.m_mask: m_mask_batch,
                     self.q: q_batch, self.q_mask: q_mask_batch, self.y: y_batch}
        return feed_dict

    def _prepro_sent_batch(self,sent_batch):
        params = self.params
        # define some dimensions
        batch_size = params.batch_size
        memory_size = params.memory_size
        sent_size = params.max_sent_size

        # initialize x, mask, and memory vectors
        x_batch = np.zeros([batch_size, memory_size, sent_size])
        x_mask_batch = np.zeros([batch_size, memory_size, sent_size])
        m_mask_batch = np.zeros([batch_size, memory_size])

        # TODO wat is dis
        for n, i, j in np.ndindex(x_batch.shape):
            if i < len(sent_batch[n]) and j < len(sent_batch[n][-i-1]):
                x_batch[n, i, j] = sent_batch[n][-i-1][j]
                x_mask_batch[n, i, j] = 1

        for n, i in np.ndindex(m_mask_batch.shape):
            if i < len(sent_batch[n]):
                m_mask_batch[n, i] = 1

        return x_batch, x_mask_batch, m_mask_batch

    def _prepro_ques_batch(self, ques_batch):
        params = self.params
        # FIXME : adhoc for now!
        # defining dimensions
        batch_size = params.batch_size
        sent_size = params.max_sent_size

        # initializing the batches
        q_batch = np.zeros([batch_size, sent_size])
        q_mask_batch = np.zeros([batch_size, sent_size])

        # TODO some kind of stuff
        for n, j in np.ndindex(q_batch.shape):
            if j < len(ques_batch[n]):
                q_batch[n, j] = ques_batch[n][j]
                q_mask_batch[n, j] = 1

        return q_batch, q_mask_batch

    def _prepro_label_batch(self, label_batch):
        # print "label_batch: ",label_batch
        return np.array(label_batch)
