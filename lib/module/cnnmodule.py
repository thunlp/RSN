import tensorflow as tf
from tensorflow.keras.layers import Conv1D, Lambda,\
  Dense, Activation, Flatten, MaxPooling1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2

import numpy as np
import numpy.random as rng

NEAR_0 = 1e-10



# loss functions
def sigmoid_cross_entropy_loss(pred,labels):
    labels_float = tf.cast(labels, tf.float32)
    loss = -tf.reduce_mean(labels_float*tf.log(pred+NEAR_0)+(1-labels_float)*tf.log(1-pred+NEAR_0))
    return loss

def denoise_loss(pred,labels,p_cond=1.0):
    labels_float = tf.cast(labels, tf.float32)
    loss = -tf.reduce_mean(labels_float*tf.log(pred+NEAR_0)+(1-labels_float)*(1+p_cond*(1-pred))*tf.log(1-pred+NEAR_0))
    return loss

def sigmoid_cond_loss(pred):
    loss = -tf.reduce_mean(pred*tf.log(pred+NEAR_0)+(1-pred)*tf.log(1-pred+NEAR_0))
    return loss

def softmax_loss_with_logits(logits, labels, n_clusters):
    labels = tf.cast(labels, tf.int32)
    one_hots = tf.cast(tf.one_hot(labels,n_clusters),dtype=tf.float32)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=one_hots))
    return loss

    
# embedding layers
class Embedding_word():
    def __init__(self,input_dim,output_dim,weights,trainable=True,name='word_embedding'):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            self.word_embedding = tf.get_variable('word_embeddings', initializer=weights, dtype=tf.float32,trainable=trainable)
            self.word_embedding = tf.concat([self.word_embedding,
                                        tf.get_variable("unk_word_embedding", [1, output_dim], dtype=tf.float32,
                                            initializer=tf.contrib.layers.xavier_initializer()),
                                        tf.constant(np.zeros((1, output_dim), dtype=np.float32))], 0)

    def __call__(self,idx_input):
        idx_input = tf.cast(idx_input,tf.int32)
        return tf.nn.embedding_lookup(self.word_embedding, idx_input)

def word_embedding(word, word_vec_mat, var_scope=None, word_embedding_dim=50, add_unk_and_blank=True):
    with tf.variable_scope(var_scope or 'word_embedding', reuse=tf.AUTO_REUSE):
        word_embedding = tf.get_variable('word_embedding', initializer=word_vec_mat, dtype=tf.float32)
        if add_unk_and_blank:
            word_embedding = tf.concat([word_embedding,
                                        tf.get_variable("unk_word_embedding", [1, word_embedding_dim], dtype=tf.float32,
                                            initializer=tf.contrib.layers.xavier_initializer()),
                                        tf.constant(np.zeros((1, word_embedding_dim), dtype=np.float32))], 0)
        x = tf.nn.embedding_lookup(word_embedding, word)
        return x

def pos_embedding(pos1, pos2, var_scope=None, pos_embedding_dim=5, max_length=120):
    with tf.variable_scope(var_scope or 'pos_embedding', reuse=tf.AUTO_REUSE):
        pos_tot = max_length * 2

        pos1_embedding = tf.get_variable('real_pos1_embedding', [pos_tot, pos_embedding_dim], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer()) 
        pos2_embedding = tf.get_variable('real_pos2_embedding', [pos_tot, pos_embedding_dim], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer()) 

        input_pos1 = tf.nn.embedding_lookup(pos1_embedding, pos1)
        input_pos2 = tf.nn.embedding_lookup(pos2_embedding, pos2)
        x = tf.concat([input_pos1, input_pos2], -1)
        return x

def word_position_embedding(word, word_vec_mat, pos1, pos2, var_scope=None, word_embedding_dim=50, pos_embedding_dim=5, max_length=120, add_unk_and_blank=True):
    w_embedding = word_embedding(word, word_vec_mat, var_scope=var_scope, word_embedding_dim=word_embedding_dim, add_unk_and_blank=add_unk_and_blank)
    p_embedding = pos_embedding(pos1, pos2, var_scope=var_scope, pos_embedding_dim=pos_embedding_dim, max_length=max_length)
    return tf.concat([w_embedding, p_embedding], -1)



# initilizers
def W_init(shape, name = None, dtype = tf.float32, partition_info = None):
    """Initialize weights as in paper"""
    values = rng.normal(loc = 0, scale = 1e-2, size = shape)
    return tf.Variable(values, name = name, dtype = dtype)

def b_init_conv(shape, name = None, dtype = tf.float32, partition_info = None):
    """Initialize bias for Conv2D layers as in paper"""
    values = rng.normal(loc = 0.5, scale = 1e-2, size = shape)
    return tf.Variable(values, name = name, dtype = dtype)

def b_init_dense(shape, name = None, dtype = tf.float32, partition_info = None):
    """Initialize bias for dense layer2 as in paper"""
    values = rng.normal(loc = 0, scale = 2e-1, size = shape)
    return tf.Variable(values, name = name, dtype = dtype)    



# cnn module
def _cnn_(cnn_input_shape,name=None):
    with tf.variable_scope(name or 'convnet', reuse=tf.AUTO_REUSE):
        convnet = Sequential()
        convnet.add(Conv1D(230, 3,
            input_shape = cnn_input_shape,
            kernel_initializer = W_init,
            bias_initializer = b_init_conv,
            kernel_regularizer=l2(2e-4)
            ))
        convnet.add(MaxPooling1D(pool_size=cnn_input_shape[0]-4))
        convnet.add(Activation('relu'))

        convnet.add(Flatten())
        convnet.add(Dense(cnn_input_shape[-1]*230, activation = 'sigmoid',
            kernel_initializer = W_init,
            bias_initializer = b_init_dense,
            kernel_regularizer=l2(1e-3)
            ))
    return convnet