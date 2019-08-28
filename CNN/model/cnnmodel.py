import tensorflow as tf
from tensorflow import keras as K
import numpy as np
import os

from module.cnnmodule import _cnn_, W_init, b_init_dense, \
softmax_loss_with_logits,Embedding_word

class CNN():
    def __init__(self, session, word_vec_mat, n_clusters = 20, max_len=120, pos_emb_dim=5, dropout=0.2):
        self.sess = session
        K.backend.set_session(self.sess)

        self.batch_shape = (3,max_len)
        self.n_clusters = n_clusters
        #defining layers
        dict_shape = word_vec_mat.shape
        self.word_emb = Embedding_word(dict_shape[0]+2, dict_shape[1], weights=word_vec_mat,trainable=True, name='word_embedding')
        self.pos1_emb = K.layers.Embedding(max_len*2, pos_emb_dim, embeddings_initializer='uniform', trainable=True, name='pos1_embedding')
        self.pos2_emb = K.layers.Embedding(max_len*2, pos_emb_dim, embeddings_initializer='uniform', trainable=True, name='pos2_embedding')
        self.drop = K.layers.Dropout(rate=dropout, name='dropout')

        cnn_input_shape = (max_len,dict_shape[1]+2*pos_emb_dim)
        self.convnet = _cnn_(cnn_input_shape, name='convnet')
        self.p = K.layers.Dense(n_clusters,kernel_initializer=W_init,bias_initializer=b_init_dense)

        #defining optimizer
        self.optimizer = tf.train.AdamOptimizer(2e-3)

    def __call__(self, idx_input, perturbation=None):
        pos1 = idx_input[:,0,:]
        pos2 = idx_input[:,1,:]
        word = idx_input[:,2,:]

        pos1_emb = self.pos1_emb(pos1)
        pos2_emb = self.pos2_emb(pos2)
        word_emb = self.word_emb(word)
        drop = self.drop(word_emb)
        if (perturbation is not None):
            drop += perturbation
        cnn_input = tf.concat([drop, pos1_emb, pos2_emb], -1)

        encoded = self.convnet(cnn_input)

        logits = self.p(encoded)
        return logits, word_emb, encoded



    def get_loss_and_emb(self, idx_input, labels):
        logits, word_emb = self(idx_input)[0:2]
        loss = softmax_loss_with_logits(logits, labels, self.n_clusters)
        return loss, word_emb
    
    def get_adv_loss(self, idx_input, labels, loss, word_emb, p_mult):
        gradient = tf.gradients(loss, [word_emb], aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N)
        p_adv = p_mult * tf.nn.l2_normalize(tf.stop_gradient(gradient), dim=1)

        logits2 = self(idx_input, p_adv)[0]
        adv_loss = softmax_loss_with_logits(logits2, labels, self.n_clusters)
        return adv_loss

    def get_v_adv_loss(self, idx_input, p_mult, power_iterations=1):
        categorical = tf.distributions.Categorical
        logits, word_emb = self(idx_input)[0:2]
        prob_dist = categorical(logits=logits)
        #generate virtual adversarial perturbation
        d = tf.random_uniform(shape=tf.shape(word_emb), dtype=tf.float32)
        for _ in range(power_iterations):
            d = (0.02) * tf.nn.l2_normalize(d, dim=1)
            p_logits = self(idx_input, d)[0]
            kl = tf.distributions.kl_divergence(prob_dist, categorical(logits=p_logits), allow_nan_stats=False)
            gradient = tf.gradients(kl, [d], aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N)
            d = tf.stop_gradient(gradient)
        d = p_mult * tf.nn.l2_normalize(d, dim=1)
        tf.stop_gradient(logits)
        #virtual adversarial loss
        p_logits = self(idx_input, d)[0]
        v_adv_losses = tf.distributions.kl_divergence(prob_dist, categorical(logits=p_logits), allow_nan_stats=False)
        return tf.reduce_mean(v_adv_losses)



    def set_ph(self,batch_size=50):
        self.batch_size = batch_size

        self.labels = tf.placeholder(tf.int32, shape=(None, 1), name='train_labels')
        self.batch_data = tf.placeholder(tf.int32, shape=(None,)+self.batch_shape, name='batch_data')

        self.logits = self(self.batch_data)[0]
        self.vectors = self(self.batch_data)[2]
        self.pred_labels = tf.cast(tf.expand_dims(tf.argmax(self.logits,axis=1),-1),tf.int32)
        self.acc = tf.reduce_mean(tf.cast(tf.equal(self.pred_labels,self.labels),tf.float32))

    def set_train_op(self,loss_type=None,p_mult=0.02):
        loss, word_emb = self.get_loss_and_emb(self.batch_data, self.labels)
        if (loss_type=='adv'):
            loss += self.get_adv_loss(self.batch_data,self.labels, loss, word_emb, p_mult)
        elif(loss_type=='v_adv'):
            loss += self.get_v_adv_loss(self.batch_data,p_mult)

        self.loss = loss
        self.label_loss = self.get_loss_and_emb(self.batch_data, self.labels)[0]
        self.opt = self.optimizer.minimize( loss )

    def init_model(self,init=None):
        self.saver = tf.train.Saver(max_to_keep=1)
        if (init is None):
            self.sess.run(tf.global_variables_initializer())
            print('model random initialized')
        else:
            ckpt = tf.train.latest_checkpoint(init)
            self.saver.restore(self.sess, ckpt)
            print('model loaded from '+init)

    def save_model(self,save_path,global_step):
        self.saver.save(self.sess,os.path.join(save_path,'CNN').replace('\\','/'),global_step=global_step)



    def train(self,dataloader, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size

        fd = {K.backend.learning_phase(): 1} #training mode
        batch_data, batch_data_label = dataloader.next_batch_cnn(batch_size)
        fd.update({self.batch_data: batch_data,self.labels: batch_data_label})

        self.sess.run([self.opt], feed_dict=fd)
        pred_labels = self.sess.run(self.pred_labels, feed_dict=fd)

    def validation(self, data, data_label):

        fd = {self.batch_data: data, self.labels: data_label, K.backend.learning_phase(): 0} #test mode        
        acc,loss = self.sess.run([self.acc,self.label_loss], feed_dict=fd)

        acc=round(float(acc),5)
        loss=round(float(loss),5)
        print('acc:',acc,'loss',loss)
        return (acc,loss)

    def pred_X(self,data):
        fd = {self.batch_data:data, K.backend.learning_phase(): 0} #test mode
        y = self.sess.run(self.pred_labels,feed_dict=fd)
        return y

    def pred_vector(self,data):
        fd = {self.batch_data:data, K.backend.learning_phase(): 0} #test mode
        vectors = self.sess.run(self.vectors,feed_dict=fd)
        return vectors