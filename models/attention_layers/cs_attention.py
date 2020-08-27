import tensorflow as tf

from tensorflow.python.keras import backend as K
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints
from keras.layers import *

class CSAttention(Layer):
    """
    Implementation of modified structured self attention as proposed 
    by https://www.aclweb.org/anthology/N19-1373.pdf

    Similar as structured self attention of https://arxiv.org/pdf/1703.03130.pdf, but
    add a forward layer of the LSTM network.

    Attention operation, with a context/query vector, for temporal data.
    Supports Masking.  by using a windowed context vector to assist the attention
    # Input shape
        4D tensor with shape: `(samples, sentence, steps, features)`.
    # Output shape
        2D tensor with shape: `(samples, sentence, features)`.
    How to use:
    Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
    The dimensions are inferred based on the output shape of the RNN.
    """

    def __init__(self, metadata, hu_utterance = 50, hu_dialogue = 100,
                 da = 350, r = 30, hu_fc = 100, embedding=360,
                 cut_gradient=False, aggregate="sum", discount=1, return_coefficients=False,
                 W_regularizer=None, W_context_regularizer=None, u_regularizer=None, b_regularizer=None,
                 W_constraint=None, W_context_constraint=None, u_constraint=None, b_constraint=None,
                 bias=True, **kwargs):

        self.metadata = metadata

        self.hu_utterance = hu_utterance
        self.hu_dialogue = hu_dialogue

        self.da = da
        self.r = r
        self.hu_fc = hu_fc

        self.embedding = embedding
        
        self.cut_gradient = cut_gradient
        self.aggregate = aggregate
        self.discount = discount
        self.supports_masking = True
        self.return_coefficients = return_coefficients
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.W_context_regularizer = regularizers.get(W_context_regularizer)
        self.u_regularizer = regularizers.get(u_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.W_context_constraint = constraints.get(W_context_constraint)
        self.u_constraint = constraints.get(u_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias

        self.LSTM = LSTM_CUSTOM(self, hu_dialogue, hu_dialogue)

        super(CSAttention, self).__init__(**kwargs)

    def build(self,input_shape ): 
        #assert len(input_shape) == 4
        

        self.W = self.add_weight(shape = (self.da, self.hu_utterance*2,),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)

        self.W_context = self.add_weight(shape = (self.da, self.hu_dialogue,),
                                         initializer=self.init,
                                         name='{}_W_context'.format(self.name),
                                         regularizer=self.W_context_regularizer,
                                         constraint=self.W_context_constraint)

        if self.bias:
            self.b = self.add_weight(shape = (self.da,),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
            
        self.u = self.add_weight(shape = (self.r,self.da),
                                 initializer=self.init,
                                 name='{}_u'.format(self.name),
                                 regularizer=self.u_regularizer,
                                 constraint=self.u_constraint)

        super(CSAttention, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):

        max_utterance_len = self.metadata['max_utterance_len']

        def compute_att(res, x_):
            
            context = res[0]
        
            uit = dot_product(x_, self.W)
            c = dot_product(context, self.W_context)
            
            if self.bias:
                uit += self.b

            uit = K.tanh(tf.add(uit, K.expand_dims(c, 1)))

   
            ait = dot_product(uit, self.u)
              
            ait = Reshape(target_shape=(self.r,x_.shape[1]))(ait)
            
    
            A = tf.keras.activations.softmax(ait, axis = -1)
            
            M = tf.keras.backend.batch_dot(A, x_)
            

            reshaped_M = Reshape((M.shape[1] * M.shape[2],))(M)
   
            fc_layer = Dense(self.hu_fc, activation='relu')(reshaped_M)
            
            context = self.LSTM.forward_pass(context,fc_layer) 
    
            return [context, Reshape(target_shape=(max_utterance_len,self.r))(A)] 

        x_t = tf.transpose(x, [1,0,2,3])
        x_t = Reshape(target_shape=(-1,max_utterance_len,self.hu_dialogue))(x_t)
    
        output, weights = tf.scan(compute_att, x_t,
                                  initializer=[K.zeros_like(x_t[0,:,0,:]),
                                              K.zeros_like(x_t[0,:,:,0:self.r])])
 
        
        output = tf.transpose(output, [1,0,2])

        if self.return_coefficients:
            return [output, weights]

        else:
            return [output]

    def compute_output_shape(self, input_shape):
        if self.return_coefficients:
            return [(input_shape[0], input_shape[1], input_shape[-1]), 
                    (input_shape[0], input_shape[1], input_shape[-1], 1)]
        else: 
            return [(input_shape[0], input_shape[1], input_shape[-1])]


def dot_product(x, kernel):
    """
    https://github.com/richliao/textClassifier/issues/13#issuecomment-377323318
    Wrapper for dot product operation, in order to be compatible with both
    Theano and Tensorflow
    Args:
        x (): input
        kernel (): weights
    Returns:
    """
    if K.backend() == 'tensorflow':
        return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
    else:
        return K.dot(x, kernel)


class LSTM_CUSTOM:
    """Implementation of a Gated Recurrent Unit (LSTM) as described in [1].
    
    [1] Chung, J., Gulcehre, C., Cho, K., & Bengio, Y. (2014). Empirical evaluation of gated recurrent neural networks on sequence modeling. arXiv preprint arXiv:1412.3555.
    
    Arguments
    ---------
    input_dimensions: int
        The size of the input vectors (x_t).
    hidden_size: int
        The size of the hidden layer vectors (h_t).
    dtype: obj
        The datatype used for the variables and constants (optional).
    """
    
    def __init__(self, layer, input_dimensions, hidden_size, dtype=tf.float32,
                 batch_size=128):
        self.batch_size =batch_size
        self.input_dimensions = input_dimensions
        self.hidden_size = hidden_size
        
        # Weights for input vectors of shape (input_dimensions, hidden_size)
        self.Wr = tf.Variable(tf.random.truncated_normal(dtype=dtype, shape=(self.input_dimensions, self.hidden_size), mean=0, stddev=0.01), name='Wr')
        self.Wz = tf.Variable(tf.random.truncated_normal(dtype=dtype, shape=(self.input_dimensions, self.hidden_size), mean=0, stddev=0.01), name='Wz')
        self.Wh = tf.Variable(tf.random.truncated_normal(dtype=dtype, shape=(self.input_dimensions, self.hidden_size), mean=0, stddev=0.01), name='Wh')
        
        # Weights for hidden vectors of shape (hidden_size, hidden_size)
        self.Ur = tf.Variable(tf.random.truncated_normal(dtype=dtype, shape=(self.hidden_size, self.hidden_size), mean=0, stddev=0.01), name='Ur')
        self.Uz = tf.Variable(tf.random.truncated_normal(dtype=dtype, shape=(self.hidden_size, self.hidden_size), mean=0, stddev=0.01), name='Uz')
        self.Uh = tf.Variable(tf.random.truncated_normal(dtype=dtype, shape=(self.hidden_size, self.hidden_size), mean=0, stddev=0.01), name='Uh')
        
        # Biases for hidden vectors of shape (hidden_size,)
        self.br = tf.Variable(tf.random.truncated_normal(dtype=dtype, shape=(self.hidden_size,), mean=0, stddev=0.01), name='br')
        self.bz = tf.Variable(tf.random.truncated_normal(dtype=dtype, shape=(self.hidden_size,), mean=0, stddev=0.01), name='bz')
        self.bh = tf.Variable(tf.random.truncated_normal(dtype=dtype, shape=(self.hidden_size,), mean=0, stddev=0.01), name='bh')
        

    def reset_h(self):
        self.h_t = tf.Variable(tf.zeros(dtype=tf.float32, shape=(self.batch_size,
                                                     self.hidden_size)),
                               trainable = False)

    def forward_pass(self, h_t, x_t):
        """Perform a forward pass.
        
        Arguments
        ---------
        h_tm1: np.matrix
            The hidden state at the previous timestep (h_{t-1}).
        x_t: np.matrix
            The input vector.
        """
        # Definitions of z_t and r_t
        z_t = tf.sigmoid(dot_product(x_t, self.Wz) + dot_product(h_t, self.Uz) + self.bz)
        r_t = tf.sigmoid(dot_product(x_t, self.Wr) + dot_product(h_t, self.Ur) + self.br)
        
        # Definition of h~_t
        h_proposal = tf.tanh(dot_product(x_t, self.Wh) +
                             dot_product(tf.multiply(r_t, h_t), self.Uh) + self.bh)
        
        # Compute the next hidden state
        h_t = tf.multiply(1 - z_t, h_t) + tf.multiply(z_t, h_proposal)
        
        return h_t