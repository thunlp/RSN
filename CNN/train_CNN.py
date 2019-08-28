import tensorflow as tf

import numpy as np
import json
import sys
import os
import argparse

sys.path.append(os.path.abspath('../lib/'))
from dataloader.dataloader import dataloader
from model.cnnmodel import CNN
from module.clusters import create_msg
from evaluation.evaluation import ClusterEvaluation
from kit.messager import messager

def train_CNN(train_data_file,test_data_file,wordvec_file,load_model_name,save_model_name,
    loss_type,max_len=120, pos_emb_dim=5,batch_size=100,batch_num=1000,epoch_num=1,val_size=1000):

    # preparing saving files
    save_path = os.path.join('model_file',save_model_name).replace('\\','/')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # train data loading
    print('-----Data Loading-----')
    dataloader_train = dataloader(train_data_file, wordvec_file, max_len=max_len)
    dataloader_test = dataloader(test_data_file, wordvec_file, max_len=max_len)
    word_emb_dim = dataloader_train._word_emb_dim_()
    word_vec_mat = dataloader_train._word_vec_mat_()
    print('word_emb_dim is {}'.format(word_emb_dim))

    # compile model
    print('-----Model Intializing-----')
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    config.gpu_options.allow_growth = True
    sess = tf.Session(config = config)    
    cnn = CNN(session=sess,word_vec_mat=word_vec_mat,max_len=max_len, pos_emb_dim=pos_emb_dim,dropout=0.2)
    cnn.set_ph(batch_size=batch_size)
    cnn.set_train_op(loss_type=loss_type,p_mult=0.02)
    cnn.init_model()

    print('-----Testing Data Preparing-----')

    # preparing testing samples
    val_testset_input, val_testset_label = dataloader_test.next_batch_cnn(val_size)
    val_trainset_input, val_trainset_label = dataloader_train.next_batch_cnn(val_size)

    # intializing parameters
    batch_num_list = [batch_num]

    for epoch in range(epoch_num):
        # preparing message lists
        msger = messager(save_path=save_path,types=['batch_num','train_acc','train_l','test_acc','test_l'], json_name='CNNmsg'+str(epoch)+'.json')

        print('------epoch {}------'.format(epoch))
        print('max batch num to train is {}'.format(batch_num_list[epoch]))
        for i in range(batch_num_list[epoch]):
            # training
            cnn.train(dataloader_train)

            # testing and saving
            if i % 10 == 0:
                print('temp_batch_num: ', i,' total_batch_num: ', batch_num_list[epoch])
            if i % 100 == 0:
                print('model_name',save_model_name)
                print('trainset:')
                val_trainset_info = cnn.validation(val_trainset_input, val_trainset_label)
                print('testset:')
                val_testset_info = cnn.validation(val_testset_input, val_testset_label)
                msger.record_message((i,)+val_trainset_info+val_testset_info)
                msger.save_json()
                cnn.save_model(save_path=save_path,global_step=i)
                print('model and messages saved.')

        # Clustering
        print('Data to cluster loading...')
        msger = messager(save_path=save_path,types=['method','F1','precision','recall','msg'], json_name='cluster_msg'+str(epoch)+'.json')
        data_to_cluster, gt = dataloader_test._data_()
        for i,item in enumerate(gt):
            gt[i]=dataloader_test.relid_dict[item]

        print('-----CNN Clustering-----')
        cluster_result = cnn.pred_X(data_to_cluster)
        cluster_result = np.squeeze(cluster_result).tolist()
        cluster_msg = create_msg(cluster_result)

        print('Evaluating...')
        cluster_eval = ClusterEvaluation(gt,cluster_result).printEvaluation()
        msger.record_message(['CNN',cluster_eval['F1'],cluster_eval['precision'],
            cluster_eval['recall'],cluster_msg])
        msger.save_json()
        print(cluster_eval)
        print('clustering messages saved.')

        print("-----End-----")
        print("The model name is:",save_model_name)
        print("loss type is:",loss_type)



if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu",type=str,default='0')
    parser.add_argument("--train_data_file",type=str,default='../data/fewrel_ori/fewrel80_test_train.json')
    parser.add_argument("--test_data_file",type=str,default='../data/fewrel_ori/fewrel80_test_test.json')
    parser.add_argument("--wordvec_file",type=str,default='../data/wordvec/word_vec.json')
    parser.add_argument("--load_model_name",type=str,default=None)
    parser.add_argument("--save_model_name",type=str,default="CNN")
    parser.add_argument("--loss_type",type=str,default='none')
    parser.add_argument("--batch_num",type=str,default=2000)
    parser.add_argument("--val_size",type=str,default=1000)
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu

    train_CNN(
        train_data_file=args.train_data_file,
        test_data_file=args.test_data_file,
        wordvec_file=args.wordvec_file,
        load_model_name=args.load_model_name,
        save_model_name=args.save_model_name,
        loss_type=args.loss_type,
        batch_num=args.batch_num,
        val_size=args.val_size)