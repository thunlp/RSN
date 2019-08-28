import tensorflow as tf
from tensorflow import keras as K
import numpy as np
import os

from module.cnnmodule import _cnn_, W_init, b_init_dense, \
sigmoid_cross_entropy_loss,denoise_loss,sigmoid_cond_loss,Embedding_word

class RSN():
    def __init__(self, session, word_vec_mat, max_len=120, pos_emb_dim=5, dropout=0.2):
        self.sess = session
        K.backend.set_session(self.sess)

        self.batch_shape = (3,max_len)
        #defining layers
        dict_shape = word_vec_mat.shape
        self.word_emb = Embedding_word(dict_shape[0]+2, dict_shape[1], weights=word_vec_mat,trainable=True, name='word_embedding')
        self.pos1_emb = K.layers.Embedding(max_len*2, pos_emb_dim, embeddings_initializer='uniform', trainable=True, name='pos1_embedding')
        self.pos2_emb = K.layers.Embedding(max_len*2, pos_emb_dim, embeddings_initializer='uniform', trainable=True, name='pos2_embedding')
        self.drop = K.layers.Dropout(rate=dropout, name='dropout')

        cnn_input_shape = (max_len,dict_shape[1]+2*pos_emb_dim)
        self.convnet = _cnn_(cnn_input_shape, name='convnet')
        self.p = K.layers.Dense(1,activation='sigmoid',kernel_initializer=W_init,bias_initializer=b_init_dense)

        #defining optimizer
        self.optimizer = tf.train.AdamOptimizer(1e-4)



    def __call__(self, left_input, right_input, left_perturbation=None, right_perturbation=None):
        left_pos1 = left_input[:,0,:]
        left_pos2 = left_input[:,1,:]
        left_word = left_input[:,2,:]
        right_pos1 = right_input[:,0,:]
        right_pos2 = right_input[:,1,:]
        right_word = right_input[:,2,:]

        left_pos1_emb = self.pos1_emb(left_pos1)
        left_pos2_emb = self.pos2_emb(left_pos2)
        left_word_emb = self.word_emb(left_word)
        left_drop = self.drop(left_word_emb)
        if (left_perturbation is not None):
            left_drop += left_perturbation
        left_cnn_input = tf.concat([left_drop, left_pos1_emb, left_pos2_emb], -1)

        right_pos1_emb = self.pos1_emb(right_pos1)
        right_pos2_emb = self.pos2_emb(right_pos2)
        right_word_emb = self.word_emb(right_word)
        right_drop = self.drop(right_word_emb)
        if (right_perturbation is not None):
            right_drop += right_perturbation
        right_cnn_input = tf.concat([right_drop, right_pos1_emb, right_pos2_emb], -1)

        encoded_l = self.convnet(left_cnn_input)
        encoded_r = self.convnet(right_cnn_input)
        L1_distance = lambda x: tf.abs(x[0]-x[1])
        both = K.layers.Lambda(L1_distance)([encoded_l,encoded_r])
        prediction = self.p(both)
        return prediction, left_word_emb, right_word_emb, encoded_l, encoded_r



    def get_loss_and_emb(self, left_input, right_input, labels):
        pred, left_word_emb, right_word_emb = self(left_input,right_input)[0:3]
        loss = sigmoid_cross_entropy_loss(pred,labels)
        return loss, left_word_emb, right_word_emb

    def get_denoise_loss(self, left_input, right_input, labels):
        pred = self(left_input,right_input)[0]
        loss = denoise_loss(pred,labels)
        return loss

    def get_cond_loss(self, left_input, right_input):
        pred = self(left_input, right_input)[0]
        cond_loss = sigmoid_cond_loss(pred)
        return cond_loss
    
    def get_adv_loss(self, left_input, right_input, labels, loss, left_word_emb, right_word_emb, p_mult):
        left_gradient,right_gradient = tf.gradients(loss, [left_word_emb,right_word_emb], 
            aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N)
        left_p_adv = p_mult * tf.nn.l2_normalize(tf.stop_gradient(left_gradient), dim=1)
        right_p_adv = p_mult * tf.nn.l2_normalize(tf.stop_gradient(right_gradient), dim=1)

        pred2 = self(left_input, right_input, left_p_adv, right_p_adv)[0]
        adv_loss = sigmoid_cross_entropy_loss(pred2, labels)
        return adv_loss
    
    def get_v_adv_loss(self, ul_left_input, ul_right_input, p_mult, power_iterations=1):
        bernoulli = tf.distributions.Bernoulli
        prob, left_word_emb, right_word_emb = self(ul_left_input, ul_right_input)[0:3]
        prob = tf.clip_by_value(prob, 1e-7, 1.-1e-7)
        prob_dist = bernoulli(probs=prob)
        #generate virtual adversarial perturbation
        left_d = tf.random_uniform(shape=tf.shape(left_word_emb), dtype=tf.float32)
        right_d = tf.random_uniform(shape=tf.shape(right_word_emb), dtype=tf.float32)
        for _ in range(power_iterations):
            left_d = (0.02) * tf.nn.l2_normalize(left_d, dim=1)
            right_d = (0.02) * tf.nn.l2_normalize(right_d, dim=1)
            p_prob = tf.clip_by_value(self(ul_left_input, ul_right_input, left_d, right_d)[0], 1e-7, 1.-1e-7)
            kl = tf.distributions.kl_divergence(prob_dist, bernoulli(probs=p_prob), allow_nan_stats=False)
            left_gradient,right_gradient = tf.gradients(kl, [left_d,right_d],
                aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N)
            left_d = tf.stop_gradient(left_gradient)
            right_d = tf.stop_gradient(right_gradient)
        left_d = p_mult * tf.nn.l2_normalize(left_d, dim=1)
        right_d = p_mult * tf.nn.l2_normalize(right_d, dim=1)
        tf.stop_gradient(prob)
        #virtual adversarial loss
        p_prob = tf.clip_by_value(self(ul_left_input, ul_right_input, left_d, right_d)[0], 1e-7, 1.-1e-7)
        v_adv_losses = tf.distributions.kl_divergence(prob_dist, bernoulli(probs=p_prob), allow_nan_stats=False)
        return tf.reduce_mean(v_adv_losses)



    def set_ph(self,batch_size=50):
        self.batch_size = batch_size

        self.labels = tf.placeholder(tf.int32, shape=(None, 1), name='train_labels')
        self.batch_left = tf.placeholder(tf.int32, shape=(None,)+self.batch_shape, name='l_batch_left')
        self.batch_right = tf.placeholder(tf.int32, shape=(None,)+self.batch_shape, name='l_batch_right')
        self.ul_batch_left = tf.placeholder(tf.int32, shape=(None,)+self.batch_shape, name='ul_batch_left')
        self.ul_batch_right = tf.placeholder(tf.int32, shape=(None,)+self.batch_shape, name='ul_batch_right')

        self.pred = self(self.batch_left, self.batch_right)[0]
        self.vector_left = self(self.batch_left, self.batch_right)[3]
        self.hard_pred = tf.cast(tf.round(self.pred),tf.int32)
        self.fp = tf.reduce_mean(tf.cast(tf.less(self.hard_pred,self.labels),tf.float32))
        self.fn = tf.reduce_mean(tf.cast(tf.greater(self.hard_pred,self.labels),tf.float32))
        self.tp = tf.reduce_mean(tf.cast(tf.equal(self.labels,0),tf.float32)*tf.cast(tf.equal(self.hard_pred,0),tf.float32))
        self.tn = tf.reduce_mean(tf.cast(tf.equal(self.labels,1),tf.float32)*tf.cast(tf.equal(self.hard_pred,1),tf.float32))

    def set_train_op(self,trainset_loss_type=None,testset_loss_type=None,p_denoise=1.0,p_cond=0.05,lambda_s=1,p_mult=0.02):
        self.load_l = True
        if (trainset_loss_type=='cross'):
            loss = self.get_loss_and_emb(self.batch_left, self.batch_right, self.labels)[0]
        elif(trainset_loss_type=='denoise'):
            loss = self.get_loss_and_emb(self.batch_left, self.batch_right, self.labels)[0]
            loss += self.get_cond_loss(self.batch_left, self.batch_right)*p_denoise
        elif(trainset_loss_type=='denoise2'):
            loss = self.get_denoise_loss(self.batch_left, self.batch_right, self.labels)
        elif (trainset_loss_type == 'adv'):
            loss, left_emb, right_emb = self.get_loss_and_emb(self.batch_left, self.batch_right, self.labels)
            loss += self.get_adv_loss(self.batch_left,self.batch_right, self.labels, loss, left_emb, right_emb, p_mult) * lambda_s
        elif (trainset_loss_type == 'v_adv'):
            loss = self.get_loss_and_emb(self.batch_left, self.batch_right, self.labels)[0]
            loss += self.get_v_adv_loss(self.batch_left,self.batch_right, p_mult) * lambda_s
        elif (trainset_loss_type == 'v_adv_denoise'):
            loss = self.get_loss_and_emb(self.batch_left, self.batch_right, self.labels)[0]
            loss += self.get_cond_loss(self.batch_left, self.batch_right)*p_denoise
            loss += self.get_v_adv_loss(self.batch_left,self.batch_right, p_mult) * lambda_s
        elif (trainset_loss_type == 'v_adv_denoise2'):
            loss = self.get_loss_and_emb(self.batch_left, self.batch_right, self.labels)[0]
            loss += self.get_cond_loss(self.batch_left, self.batch_right)*p_denoise
        else:
            loss = []
            self.load_l = False

        self.load_ul = True
        if (testset_loss_type =='cond'):
            loss += self.get_cond_loss(self.ul_batch_left, self.ul_batch_right) * p_cond
        elif (testset_loss_type =='v_adv'):
            loss += self.get_cond_loss(self.ul_batch_left, self.ul_batch_right) * p_cond
            loss += self.get_v_adv_loss(self.ul_batch_left,self.ul_batch_right, p_mult) * lambda_s * p_cond
        elif (testset_loss_type =='v_adv_alone'):
            loss += self.get_v_adv_loss(self.ul_batch_left,self.ul_batch_right, p_mult) * lambda_s * p_cond
        else:
            self.load_ul = False

        self.loss = loss
        self.opt = self.optimizer.minimize( loss )

        loss, left_emb, right_emb = self.get_loss_and_emb(self.batch_left, self.batch_right, self.labels)
        self.label_loss = loss
        self.label_adv_loss = self.get_adv_loss(self.batch_left,self.batch_right, self.labels, loss, left_emb, right_emb, p_mult)


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
        self.saver.save(self.sess,os.path.join(save_path,'RSN').replace('\\','/'),global_step=global_step)



    def train(self,dataloader_trainset,dataloader_testset, batch_size=None, same_ratio=0.5):
        if batch_size is None:
            batch_size = self.batch_size

        fd = {K.backend.learning_phase(): 1} #training mode
        if (self.load_l):
            batch_data_left, batch_data_right, batch_data_label = dataloader_trainset.next_batch(batch_size, same_ratio = same_ratio)
            fd.update({self.batch_left: batch_data_left,self.batch_right: batch_data_right, self.labels: batch_data_label})
        if (self.load_ul):
            ul_batch_data_left, ul_batch_data_right = dataloader_testset.next_batch_ul(batch_size)
            fd.update( {self.ul_batch_left: ul_batch_data_left, self.ul_batch_right: ul_batch_data_right} )
            
        self.sess.run([self.opt], feed_dict=fd)

    def train_unsup(self,dataloader_trainset,dataloader_testset,batch_size=None, same_ratio=0.5):
        if batch_size is None:
            batch_size = self.batch_size

        fd = {K.backend.learning_phase(): 1} #training mode
        if (self.load_l):
            batch_data_left, batch_data_right, batch_data_label = dataloader_trainset.next_batch_self(batch_size, same_ratio = same_ratio)
            fd.update({self.batch_left: batch_data_left,self.batch_right: batch_data_right, self.labels: batch_data_label})
        if (self.load_ul):
            ul_batch_data_left, ul_batch_data_right = dataloader_testset.next_batch_ul(batch_size)
            fd.update( {self.ul_batch_left: ul_batch_data_left, self.ul_batch_right: ul_batch_data_right} )
            
        self.sess.run([self.opt], feed_dict=fd)

    def validation(self, data_left, data_right, data_label):

        fd = {self.batch_left: data_left, self.batch_right: data_right,
        self.labels: data_label, K.backend.learning_phase(): 0} #test mode        
        fp,fn,tp,tn,label_loss,label_adv_loss = self.sess.run(
            [self.fp,self.fn,self.tp,self.tn,self.label_loss,self.label_adv_loss], feed_dict=fd)

        tp=round(float(tp),5)
        fp=round(float(fp),5)
        fn=round(float(fn),5)
        tn=round(float(tn),5)
        label_loss=round(float(label_loss),5)
        print('tp:',tp,'fp',fp,'fn',fn,'tn',tn,'label_loss',label_loss,'label_adv_loss',label_adv_loss)
        return (tp,fp,fn,tn,label_loss)

    def pred_X(self,data_left,data_right):
        fd = {self.batch_left:data_left, self.batch_right: data_right, K.backend.learning_phase(): 0} #test mode
        y = self.sess.run(self.pred,feed_dict=fd)
        return y

    def pred_vector(self,data):
        fd = {self.batch_left:np.array(data),K.backend.learning_phase(): 0}#test mode
        vectors = self.sess.run(self.vector_left,feed_dict=fd)
        return vectors