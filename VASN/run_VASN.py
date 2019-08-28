import tensorflow as tf

import numpy as np
import json
import sys
import os
import argparse

sys.path.append(os.path.abspath('../lib/'))
from dataloader.dataloader import dataloader
from model.siamodel import VASN
from module.clusters import *
from evaluation.evaluation import ClusterEvaluation
from kit.messager import messager

def train_SN(test_data_file,wordvec_file,load_model_name,save_result_name,
    max_len=120, pos_emb_dim=5,select_cluster='Louvain'):

    # preparing saving files
    load_path = os.path.join('model_file',load_model_name).replace('\\','/')
    save_path = os.path.join(load_path,save_result_name).replace('\\','/')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    msger = messager(save_path=save_path,types=['test_data_file'], json_name='test_msg.json')
    msger.record_message([test_data_file])
    msger.save_json()

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

    # Clustering
    print('Data to cluster loading...')
    msger = messager(save_path=save_path,types=['method','F1','precision','recall','msg'], json_name='cluster_msg.json')
    data_to_cluster, gt = dataloader_test._data_()

    if 'Louvain' in select_cluster:
        print('-----Louvain Clustering-----')
        cluster_result, cluster_msg = Louvain_no_isolation(dataset=data_to_cluster,edge_measure=SN.pred_X)

        print('Evaluating...')
        cluster_eval = ClusterEvaluation(gt,cluster_result).printEvaluation()
        msger.record_message(['Louvain',cluster_eval['F1'],cluster_eval['precision'],
            cluster_eval['recall'],cluster_msg])
        msger.save_json()
        print(cluster_eval)
        print('clustering messages saved.')


    if 'HAC' in select_cluster:
        print('-----HAC Clustering-----')
        cluster_result, cluster_msg = complete_HAC(dataset=data_to_cluster,HAC_dist=SN.pred_X,k=15)

        print('Evaluating...')
        cluster_eval = ClusterEvaluation(gt,cluster_result).printEvaluation()
        msger.record_message(['HAC',cluster_eval['F1'],cluster_eval['precision'],
            cluster_eval['recall'],cluster_msg])
        msger.save_json()
        print(cluster_eval)
        print('clustering messages saved.')

    print('End: The model is:',load_model_name)



if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu",type=str,default='0')
    parser.add_argument("--test_data_file",type=str,default='../data/nyt10/nyt10_wholeset_test.json')
    parser.add_argument("--wordvec_file",type=str,default='../data/wordvec/word_vec.json')
    parser.add_argument("--load_model_name",type=str,default='ori/cross64_full')
    parser.add_argument("--save_result_name",type=str,default='transfer_nyt10_wholeset_test_2')
    parser.add_argument("--select_cluster",type=int,default=2)
    args = parser.parse_args()
    cluster_dict={0:[],1:['Louvain'],2:['HAC'],3:['Louvain','HAC']}
    args.select_cluster=cluster_dict[args.select_cluster]

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu

    train_SN(
        test_data_file = args.test_data_file,
        wordvec_file = args.wordvec_file,
        load_model_name=args.load_model_name,
        select_cluster=args.select_cluster,
        save_result_name=args.save_result_name)