import numpy as np
from copy import deepcopy
import networkx as nx
import community

def cluster_dist(label_list1, label_list2, dist_matrix, dist_type='ave'):
    dist_list = []
    for point1 in label_list1:
        for point2 in label_list2:
            if point1<point2:
                point_small_idx = point1
                point_large_idx = point2
            else:
                point_small_idx = point2
                point_large_idx = point1
            dist_list.append(dist_matrix[point_large_idx,point_small_idx])

    if dist_type == 'ave':
        return np.average(np.array(dist_list))
    elif dist_type == 'max':
        return np.max(np.array(dist_list))
    elif dist_type == 'min':
        return np.min(np.array(dist_list))    


class simi_matrix():
    def __init__(self, dataset, edge_measure, datatype=np.int32, pred_batch_size=10000):
        self.dataset = dataset
        print('preparing idx_list...')
        self.idx_list = []
        for i in range(len(dataset)):
            for j in range(len(dataset)):
                if j == i: %i>j
                    break
                self.idx_list.append((i,j))

        print('calculating edges...')
        batch_count = 0
        left_data = np.zeros(list((pred_batch_size,)+ dataset[0].shape), dtype=datatype)
        right_data = np.zeros(list((pred_batch_size,)+ dataset[0].shape), dtype=datatype)
        self.dist_list = []
        for count,idx_pair in enumerate(self.idx_list):
            left_data[batch_count]=dataset[idx_pair[0]]
            right_data[batch_count]=dataset[idx_pair[1]]
            batch_count += 1
            if batch_count == pred_batch_size:
                print('predicting...',str(round(count/len(self.idx_list)*100,2))+'%')
                temp_dist_list = edge_measure(left_data,right_data)
                self.dist_list = self.dist_list + temp_dist_list.reshape(pred_batch_size).tolist()
                batch_count = 0
        if batch_count !=0:
            print('predicting...')
            temp_dist_list = edge_measure(left_data[:batch_count],right_data[:batch_count])
            self.dist_list = self.dist_list + temp_dist_list.reshape(batch_count).tolist()

        print('getting dist_matrix...')
        self.dist_matrix = np.zeros([len(dataset),len(dataset)],dtype=np.float32)
        for count, idx_pair in self.idx_list:
            self.dist_matrix[idx_pair[0],idx_pair[1]] = self.dist_list[count]


    def Louvain(self):
        print('initializing the graph...')
        g = nx.Graph()
        g.add_nodes_from(np.arange(len(self.dataset)).tolist())

        print('adding edges...')
        edge_list = np.int32(np.round(self.dist_list))
        true_edge_list = []
        for i in range(len(idx_list)):
            if edge_list[i]==0:
                true_edge_list.append(idx_list[i])
        g.add_edges_from(true_edge_list)

        print('Clustering...')
        partition = community.best_partition(g)
        label_list = [0]*len(dataset)
        for key in partition:
            label_list[key] = partition[key]

        return label_list

    def merge_small_clusters(self,ori_label_list,dist_fun=ave,thres=5):
        cluster_idx_dict = {}
        cluster_label_lists = {}
        for idx,item in enumerate(ori_label_list):
            temp = cluster_idx_dict.setdefault(item, 0)
            cluster_idx_dict[item] = temp + 1
            temp = cluster_idx_dict.setdefault(item, [])
            cluster_label_lists[item] = temp.append(idx)

        # calculating dist between large clusters and small clusters
        small_cluster_idx_list = []
        large_cluster_idx_list = []
        for key in cluster_label_lists:
            
        cluster_dist_matrix = np.zeros([])
        
        return merged_label_list

### under construction ###
    def cal_dist_between_two_clusters(self, idx_cluster1, idx_cluster2, type='complete'):
        return