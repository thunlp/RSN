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
from kit.visualizer import plot

def visualize_CNN(test_data_file,wordvec_file,model_file,save_figure_name,
    max_len=120, pos_emb_dim=5):

    # preparing saving files
    save_path = os.path.join('figure',save_figure_name).replace('\\','/')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    msger = messager(save_path=save_path,types=['model_file','test_data_file'],
        json_name='para_msg.json')
    msger.record_message([model_file,test_data_file])
    msger.save_json()

    # train data loading
    print('-----Data Loading-----')
    dataloader_test = dataloader(test_data_file, wordvec_file, max_len=max_len)
    word_emb_dim = dataloader_test._word_emb_dim_()
    word_vec_mat = dataloader_test._word_vec_mat_()

    # compile model
    print('-----Model Intializing-----')
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    config.gpu_options.allow_growth = True
    sess = tf.Session(config = config)    
    cnn = CNN(session=sess,word_vec_mat=word_vec_mat,max_len=max_len, pos_emb_dim=pos_emb_dim,dropout=0.2)
    cnn.set_ph(batch_size=50)
    cnn.set_train_op(loss_type='cross',p_mult=0.02)
    cnn.init_model(model_file)

    # Clustering
    print('Data to cluster loading...')
    msger = messager(save_path=save_path,types=['method','F1','precision','recall','msg'], json_name='cluster_msg.json')
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

    print('Ploting...')
    vectors = cnn.pred_vector(data=data_to_cluster)
    for i in range(len(gt)):
        gt[i]+=68
    plot(vectors, gt, save_figure_file = save_path+'/'+save_figure_name+'.png', method ='tsne')



if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu",type=str,default='0')
    parser.add_argument("--test_data_file",type=str,default='../data/fewrel_ori/fewrel80_visual68697071.json')
    parser.add_argument("--wordvec_file",type=str,default='../data/wordvec/word_vec.json')
    parser.add_argument("--model_file",type=str,default='model_file/CNN_ori')
    parser.add_argument("--save_figure_name",type=str,default='CNN_4cluster')
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu

    visualize_CNN(
        test_data_file=args.test_data_file,
        wordvec_file=args.wordvec_file,
        model_file=args.model_file,
        save_figure_name=args.save_figure_name)