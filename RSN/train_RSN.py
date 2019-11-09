import tensorflow as tf

import numpy as np
import json
import sys
import os
import argparse

sys.path.append(os.path.abspath('../lib/'))
from dataloader.dataloader import dataloader
from model.siamodel import RSN
from module.clusters import *
from evaluation.evaluation import ClusterEvaluation
from kit.messager import messager

def train_SN(train_data_file,val_data_file,test_data_file,wordvec_file,load_model_name=None,save_model_name='SN',
    trainset_loss_type='cross',testset_loss_type='none',testset_loss_mask_epoch=3,p_cond=0.03,p_denoise=1.0,
    max_len=120, pos_emb_dim=5,same_ratio=0.06,batch_size=100,batch_num=100000,epoch_num=1,
    val_size=10000,select_cluster='Louvain',omit_relid=None,labeled_sample_num=None):

    # preparing saving files
    if load_model_name is not None:
        load_path = os.path.join('model_file',load_model_name).replace('\\','/')
    else:
        load_path = None
    save_path = os.path.join('model_file',save_model_name).replace('\\','/')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    msger = messager(save_path=save_path,types=['train_data_file','val_data_file','test_data_file','load_model_name','save_model_name',
        'trainset_loss_type','testset_loss_type','testset_loss_mask_epoch','p_cond','p_denoise','same_ratio','labeled_sample_num'],
        json_name='train_msg.json')
    msger.record_message([train_data_file,val_data_file,test_data_file,load_model_name,save_model_name,
        trainset_loss_type,testset_loss_type,testset_loss_mask_epoch,p_cond,p_denoise,same_ratio,labeled_sample_num])
    msger.save_json()

    # train data loading
    print('-----Data Loading-----')
    dataloader_train = dataloader(train_data_file, wordvec_file, max_len=max_len)
    if omit_relid is not None and omit_relid>=4:
        dataloader_train.select_relation(np.arange(2,omit_relid+1,1).tolist())
    if labeled_sample_num is not None:
        dataloader_train.select_sample_num(labeled_sample_num)
    dataloader_testset = dataloader(val_data_file, wordvec_file, max_len=max_len)
    dataloader_test = dataloader(test_data_file, wordvec_file, max_len=max_len)
    word_emb_dim = dataloader_train._word_emb_dim_()
    word_vec_mat = dataloader_train._word_vec_mat_()
    print('word_emb_dim is {}'.format(word_emb_dim))

    # compile model
    print('-----Model Intializing-----')
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config = config)
    SN = RSN(session=sess,word_vec_mat=word_vec_mat,max_len=max_len, pos_emb_dim=pos_emb_dim,dropout=0.2)
    SN.set_ph(batch_size=batch_size)
    SN.set_train_op(trainset_loss_type=trainset_loss_type,testset_loss_type=testset_loss_type,p_cond=p_cond,p_denoise=p_denoise,p_mult=0.02)
    SN.init_model(load_path)

    print('-----Testing Data Preparing-----')

    # preparing testing samples
    val_testset_left_input, val_testset_right_input, val_testset_data_label = \
    dataloader_testset.next_batch(val_size,same_ratio=same_ratio)
    val_trainset_left_input, val_trainset_right_input, val_trainset_data_label = \
    dataloader_train.next_batch(val_size,same_ratio=same_ratio)

    # intializing parameters
    batch_num_list = [batch_num]*epoch_num
    clustering_test_time = np.arange(19999,batch_num,20000).tolist()
    msger_cluster = messager(save_path=save_path,types=['method','temp_batch_num','F1','precision','recall','msg'],
        json_name='cluster_msg.json')

    for epoch in range(epoch_num):
        if epoch<testset_loss_mask_epoch:
            SN.set_train_op(trainset_loss_type=trainset_loss_type,testset_loss_type='none',p_cond=p_cond,p_denoise=p_denoise,p_mult=0.02)
        else:
            SN.set_train_op(trainset_loss_type=trainset_loss_type,testset_loss_type=testset_loss_type,p_cond=p_cond,p_denoise=p_denoise,p_mult=0.02)

        # preparing message lists
        msger = messager(save_path=save_path,types=['batch_num','train_tp','train_fp','train_fn','train_tn','train_l',
            'test_tp','test_fp','test_fn','test_tn','test_l'], json_name='SNmsg'+str(epoch)+'.json')

        data_to_cluster, gt = dataloader_test._data_()

        print('------epoch {}------'.format(epoch))
        print('max batch num to train is {}'.format(batch_num_list[epoch]))
        for i in range(batch_num_list[epoch]):
            # training
            if omit_relid is not None and omit_relid == 0:
                SN.train_unsup(dataloader_train,dataloader_testset,batch_size=batch_size, same_ratio=same_ratio)
            else:
                SN.train(dataloader_train,dataloader_testset,batch_size=batch_size, same_ratio=same_ratio)

            # testing and saving
            if i % 100 == 0:
                print('temp_batch_num: ', i,' total_batch_num: ', batch_num_list[epoch])
            if i % 1000 == 0:
                print(save_model_name,'epoch:',epoch)
                print('trainset:')
                val_trainset_info = SN.validation(val_trainset_left_input, val_trainset_right_input, val_trainset_data_label)
                print('testset:')
                val_testset_info = SN.validation(val_testset_left_input, val_testset_right_input, val_testset_data_label)
                msger.record_message((i,)+val_trainset_info+val_testset_info)
                msger.save_json()
                SN.save_model(save_path=save_path,global_step=i)
                print('model and messages saved.')
            if i in clustering_test_time or i==batch_num_list[epoch]-1:
                if 'Louvain' in select_cluster:
                    print('-----Louvain Clustering-----')
                    cluster_result, cluster_msg = Louvain_no_isolation(dataset=data_to_cluster,edge_measure=SN.pred_X)
                    cluster_eval = ClusterEvaluation(gt,cluster_result).printEvaluation()
                    msger_cluster.record_message(['Louvain',i,cluster_eval['F1'],cluster_eval['precision'],
                        cluster_eval['recall'],cluster_msg])
                    msger_cluster.save_json()
                    print(cluster_eval)
                    print('clustering messages saved.')

                if 'HAC' in select_cluster:
                    print('-----HAC Clustering-----')
                    cluster_result, cluster_msg = complete_HAC(dataset=data_to_cluster,HAC_dist=SN.pred_X,k=len(list(set(gt))))
                    cluster_eval = ClusterEvaluation(gt,cluster_result).printEvaluation()
                    msger_cluster.record_message(['HAC',i,cluster_eval['F1'],cluster_eval['precision'],
                        cluster_eval['recall'],cluster_msg])
                    msger_cluster.save_json()
                    print(cluster_eval)
                    print('clustering messages saved.')

        print('End: The model is:',save_model_name, trainset_loss_type, testset_loss_type,'p_cond is:',p_cond)



if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu",type=str,default='0')
    parser.add_argument("--dataset",type=str,default='ori')
    parser.add_argument("--train_data_file",type=str,default='../data-bin/fewrel_ori/fewrel80_train.json')
    parser.add_argument("--val_data_file",type=str,default='../data-bin/fewrel_ori/fewrel80_test_train.json')
    parser.add_argument("--test_data_file",type=str,default='../data-bin/fewrel_ori/fewrel80_test_test.json')
    parser.add_argument("--wordvec_file",type=str,default='../data-bin/wordvec/word_vec.json')
    parser.add_argument("--load_model_name",type=str,default=None)
    parser.add_argument("--save_model_name",type=str,default='ori/')
    parser.add_argument("--select_cluster",type=int,default=1)
    parser.add_argument("--trainset_loss_type",type=str,default='v_adv')
    parser.add_argument("--testset_loss_type",type=str,default='v_adv')
    parser.add_argument("--testset_loss_mask_epoch",type=int,default=0)
    parser.add_argument("--p_cond",type=float,default=0.03)
    parser.add_argument("--p_denoise",type=float,default=1.0)
    parser.add_argument("--same_ratio",type=float,default=0.06)
    parser.add_argument("--batch_num",type=int,default=10000)
    parser.add_argument("--epoch_num",type=int,default=5)
    parser.add_argument("--val_size",type=int,default=10000)
    parser.add_argument("--omit_relid",type=int,default=None,help=
        "None means not omit; 0 means unsupervised mode; otherwise means reserving all the relations with relid<=omit_relid from trainset")
    parser.add_argument("--labeled_sample_num",type=int,default=None)
    args = parser.parse_args()
    cluster_dict={0:[],1:['Louvain'],2:['HAC'],3:['Louvain','HAC']}
    args.select_cluster=cluster_dict[args.select_cluster]

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu

    if args.dataset == 'ori':
        args.train_data_file = '../data-bin/fewrel_ori/fewrel80_train.json'
        args.val_data_file =  '../data-bin/fewrel_ori/fewrel80_test_train.json'
        args.test_data_file = '../data-bin/fewrel_ori/fewrel80_test_test.json'
    # elif args.dataset =='distant':
    #     args.train_data_file = '../data-bin/fewrel_distant/fewrel80_distant_train.json'
    #     args.val_data_file = '../data-bin/fewrel_distant/fewrel80_distant_test_omit.json'
    #     args.test_data_file = '../data-bin/fewrel_distant/fewrel80_test_test.json'
    else:
        raise Exception('currently only fewrel80 is available')


    train_SN(
        train_data_file = args.train_data_file,
        val_data_file = args.val_data_file,
        test_data_file = args.test_data_file,
        wordvec_file = args.wordvec_file,
        load_model_name=args.load_model_name,
        save_model_name = args.save_model_name,
        select_cluster=args.select_cluster,
        trainset_loss_type=args.trainset_loss_type,
        testset_loss_type=args.testset_loss_type,
        testset_loss_mask_epoch=args.testset_loss_mask_epoch,
        p_cond=args.p_cond,
        p_denoise=args.p_denoise,
        same_ratio=args.same_ratio,
        batch_num=args.batch_num,
        epoch_num=args.epoch_num,
        val_size=args.val_size,
        omit_relid=args.omit_relid,
        labeled_sample_num=args.labeled_sample_num)