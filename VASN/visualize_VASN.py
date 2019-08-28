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
from kit.visualizer import plot

def visualize_VASN(model_file,test_data_file,wordvec_file,save_figure_name,
    select_cluster, max_len = 120, pos_emb_dim = 5):

    # preparing saving files
    save_path = os.path.join('figure',save_figure_name).replace('\\','/')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    msger = messager(save_path=save_path,types=['model_file','test_data_file',
        'select_cluster'], json_name='para_msg.json')
    msger.record_message([model_file,test_data_file,select_cluster])
    msger.save_json()

    # data loading
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
    SN.init_model(model_file)

    # Clustering
    print('Data to cluster loading...')
    msger = messager(save_path=save_path,types=['method','F1','precision','recall','msg'], json_name='cluster_msg.json')
    data_to_cluster, gt = dataloader_test._data_()

    # if 'Louvain' in select_cluster:
    #     print('-----Louvain Clustering-----')
    #     cluster_result, cluster_msg = Louvain(dataset=data_to_cluster,edge_measure=SN.pred_X)

    #     print('Evaluating...')
    #     cluster_eval = ClusterEvaluation(gt,cluster_result).printEvaluation()
    #     msger.record_message(['Louvain',cluster_eval['F1'],cluster_eval['precision'],
    #         cluster_eval['recall'],cluster_msg])
    #     msger.save_json()
    #     print(cluster_eval)
    #     print('clustering messages saved.')


    # if 'HAC' in select_cluster:
    #     print('-----HAC Clustering-----')
    #     cluster_result, cluster_msg = complete_HAC(dataset=data_to_cluster,HAC_dist=SN.pred_X,k=len(list(set(gt))))

    #     print('Evaluating...')
    #     cluster_eval = ClusterEvaluation(gt,cluster_result).printEvaluation()
    #     msger.record_message(['HAC',cluster_eval['F1'],cluster_eval['precision'],
    #         cluster_eval['recall'],cluster_msg])
    #     msger.save_json()
    #     print(cluster_eval)
    #     print('clustering messages saved.')

    print('Ploting...')
    vectors = SN.pred_vector(data=data_to_cluster)
    plot(vectors, gt, save_figure_file = save_path+'/'+save_figure_name+'.png', method ='tsne')

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu",type=str,default='6')
    parser.add_argument("--model_file",type=str,default='model_file/ori/CEVAT64_full')
    parser.add_argument("--test_data_file",type=str,default='../data/fewrel_ori/fewrel80_visual68697071.json')
    parser.add_argument("--wordvec_file",type=str,default='../data/wordvec/word_vec.json')
    parser.add_argument("--select_cluster",type=int,default=1)
    parser.add_argument("--save_figure_name",type=str,default='CEVAT_4cluster')

    args = parser.parse_args()
    cluster_dict={1:['Louvain'],2:['HAC']}
    args.select_cluster=cluster_dict[args.select_cluster]
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu

    visualize_VASN(
        model_file = args.model_file,
        test_data_file = args.test_data_file,
        wordvec_file = args.wordvec_file,
        save_figure_name = args.save_figure_name,
        select_cluster = args.select_cluster,)