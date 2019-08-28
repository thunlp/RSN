import tensorflow as tf

import numpy as np
import json
import sys
import os
import argparse

sys.path.append(os.path.abspath('../lib/'))
from dataloader.dataloader import dataloader
from model.siamodel import VASN

def train_SN(test_data_file,wordvec_file,load_model_name,
    max_len=120, pos_emb_dim=5):

    # preparing saving files
    load_path = os.path.join('model_file',load_model_name).replace('\\','/')

    # train data loading
    print('-----Data Loading-----')
    dataloader_test = dataloader(test_data_file, wordvec_file, max_len=max_len)
    word_emb_dim = dataloader_test._word_emb_dim_()
    word_vec_mat = dataloader_test._word_vec_mat_()
    print('word_emb_dim is {}'.format(word_emb_dim))

    # compile model
    print('-----Model Intializing-----')
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 1.0
    config.gpu_options.allow_growth = True
    sess = tf.Session(config = config)
    SN = VASN(session=sess,word_vec_mat=word_vec_mat,max_len=max_len, pos_emb_dim=pos_emb_dim,dropout=0.2)
    SN.set_ph(batch_size=100)
    SN.set_train_op(trainset_loss_type='cross',testset_loss_type='none',p_mult=0.02)
    SN.init_model(init=load_path)

    print('-----Predicting-----')
    val_testset_left_input, val_testset_right_input, val_testset_data_label = \
    dataloader_testset.next_batch(batch_size = 20 ,same_ratio=0.5)

    predict_result = sn.pred_x(val_testset_left_input, val_testset_right_input)

    for i in range(20):
        print(i,'true_dist:',val_testset_data_label[i],'predict_dist:'predict_result[i])


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu",type=str,default='0')
    parser.add_argument("--test_data_file",type=str,default='../data/fewrel/testset_test.json')
    parser.add_argument("--wordvec_file",type=str,default='../data/wordvec/word_vec.json')
    parser.add_argument("--load_model_name",type=str,default=None)
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu

    train_SN(
        test_data_file = args.test_data_file,
        wordvec_file = args.wordvec_file,
        load_model_name=args.load_model_name)