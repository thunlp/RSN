import numpy as np
from copy import deepcopy
import networkx as nx
import community



# def L2_dist(left_data, right_data):
#     data_minus = np.array(left_data)-np.array(right_data)
#     data_dist = np.sum(data_minus*data_minus,1)
#     return data_dist

# find the closest two classes
def find_close(M):
    s_index, l_index = 0, 0
    min_list = np.zeros([len(M)],dtype=np.float32)
    min_index_list = np.zeros([len(M)],dtype=np.int32)
    for i,item in enumerate(M):
        if len(item):
            temp_min = min(item)
            min_list[i] = temp_min
            min_index_list[i] = item.index(temp_min)
        else:
            min_list[i] = 10000
    l_index = int(np.where(min_list==np.min(min_list))[0][0])
    s_index = min_index_list[l_index]
    return s_index, l_index # s_index < l_index

# model
def complete_HAC(dataset, HAC_dist, k, datatype=np.int32):
    #initialize C and M, C is a list of clusters, M is a list as dist_matrix

    print('the len of dataset to cluster is:'+str(len(dataset)))
    print('initializing...')
    idx_C, M, idxM = [], [], []
    for i,item in enumerate(dataset):
        idx_Ci = [i]
        idx_C.append(idx_Ci)

    print('initializing dist_matrix...')
    print('preparing idx_list...')
    idx_list = []
    for i in range(len(idx_C)):
        for j in range(len(idx_C)):
            if j == i:
                break
            idx_list.append([i,j])

    print('calculating dist_list...')
    batch_count = 0
    batch_size = 10000
    left_data = np.zeros(list((batch_size,)+ dataset[0].shape), dtype=datatype)
    right_data = np.zeros(list((batch_size,)+ dataset[0].shape), dtype=datatype)
    dist_list = []
    for count,idx_pair in enumerate(idx_list):
        left_data[batch_count]=dataset[idx_pair[0]]
        right_data[batch_count]=dataset[idx_pair[1]]
        batch_count += 1
        if batch_count == batch_size:
            print('predicting',str(round(count/len(idx_list)*100,2))+'%')
            temp_dist_list = HAC_dist(left_data,right_data)
            dist_list = dist_list + temp_dist_list.reshape(batch_size).tolist()
            batch_count = 0
    if batch_count !=0:
        print('predicting...')
        temp_dist_list = HAC_dist(left_data[:batch_count],right_data[:batch_count])
        dist_list = dist_list + temp_dist_list.reshape(batch_count).tolist()

    print('preparing dist_matrix...')
    count = 0
    for i in range(len(idx_C)):
        Mi = []
        for j in range(len(idx_C)):
            if j == i:
                break
            Mi.append(dist_list[count])
            count += 1
        M.append(Mi)

    # combine two classes
    q = len(idx_C)
    while q > k:
        s_index, l_index = find_close(M)
        idx_C[s_index].extend(idx_C[l_index])
        del idx_C[l_index]

        M_next = deepcopy(M[:-1])
        for i in range(len(idx_C)):
            for j in range(len(idx_C)):
                if j == i:
                    break

                i_old, j_old = i, j
                if i >= l_index:
                    i_old = i + 1
                if j >= l_index:
                    j_old = j + 1

                if i != s_index and j != s_index:
                    M_next[i][j]=M[i_old][j_old]
                elif i == s_index:
                    M_next[i][j]=max(M[s_index][j_old],M[l_index][j_old])
                elif j == s_index:
                    if i_old<l_index:
                        M_next[i][j]=max(M[i_old][s_index],M[l_index][i_old])
                    elif i_old>l_index:
                        M_next[i][j]=max(M[i_old][s_index],M[i_old][l_index])
        q -= 1
        print('temp cluster num is:',q,',',s_index,'and',l_index,'are combined, metric is:',M[l_index][s_index])
        M = M_next

    # decode to get label_list
    label_list = [0]*len(dataset)
    for label, temp_cluster in enumerate(idx_C):
        for idx in temp_cluster:
            label_list[idx] = label

    return label_list, create_msg(label_list)



def Louvain(dataset, edge_measure, datatype=np.int32):

    print('initializing the graph...')
    g = nx.Graph()
    g.add_nodes_from(np.arange(len(dataset)).tolist())
    print('preparing idx_list...')
    idx_list = []
    for i in range(len(dataset)):
        for j in range(len(dataset)):
            if j == i:
                break
            idx_list.append((i,j))

    print('calculating edges...')
    batch_count = 0
    batch_size = 10000
    left_data = np.zeros(list((batch_size,)+ dataset[0].shape), dtype=datatype)
    right_data = np.zeros(list((batch_size,)+ dataset[0].shape), dtype=datatype)
    edge_list = []
    for count,idx_pair in enumerate(idx_list):
        left_data[batch_count]=dataset[idx_pair[0]]
        right_data[batch_count]=dataset[idx_pair[1]]
        batch_count += 1
        if batch_count == batch_size:
            print('predicting...',str(round(count/len(idx_list)*100,2))+'%')
            temp_edge_list = edge_measure(left_data,right_data)
            edge_list = edge_list + temp_edge_list.reshape(batch_size).tolist()
            batch_count = 0
    if batch_count !=0:
        print('predicting...')
        temp_edge_list = edge_measure(left_data[:batch_count],right_data[:batch_count])
        edge_list = edge_list + temp_edge_list.reshape(batch_count).tolist()
    edge_list = np.int32(np.round(edge_list))

    print('adding edges...')
    true_edge_list = []
    for i in range(len(idx_list)):
        if edge_list[i]==0:
            true_edge_list.append(idx_list[i])
    g.add_edges_from(true_edge_list)

    print('Clustering...')
    partition = community.best_partition(g)

    # decode to get label_list
    print('decoding to get label_list...')
    label_list = [0]*len(dataset)
    for key in partition:
        label_list[key] = partition[key]

    return label_list, create_msg(label_list)

def Louvain_no_isolation(dataset, edge_measure, datatype=np.int32, iso_thres=5):

    print('initializing the graph...')
    g = nx.Graph()
    g.add_nodes_from(np.arange(len(dataset)).tolist())
    print('preparing idx_list...')
    idx_list = []
    for i in range(len(dataset)):
        for j in range(len(dataset)):
            if j == i:
                break
            idx_list.append((i,j))

    print('calculating edges...')
    batch_count = 0
    batch_size = 10000
    left_data = np.zeros(list((batch_size,)+ dataset[0].shape), dtype=datatype)
    right_data = np.zeros(list((batch_size,)+ dataset[0].shape), dtype=datatype)
    edge_list = []
    for count,idx_pair in enumerate(idx_list):
        left_data[batch_count]=dataset[idx_pair[0]]
        right_data[batch_count]=dataset[idx_pair[1]]
        batch_count += 1
        if batch_count == batch_size:
            print('predicting...',str(round(count/len(idx_list)*100,2))+'%')
            temp_edge_list = edge_measure(left_data,right_data)
            edge_list = edge_list + temp_edge_list.reshape(batch_size).tolist()
            batch_count = 0
    if batch_count !=0:
        print('predicting...')
        temp_edge_list = edge_measure(left_data[:batch_count],right_data[:batch_count])
        edge_list = edge_list + temp_edge_list.reshape(batch_count).tolist()
    simi_list = edge_list
    edge_list = np.int32(np.round(edge_list))

    #------------------
    print('forming simi_matrix...')
    simi_matrix = np.zeros([len(dataset),len(dataset)])
    for count,idx_pair in enumerate(idx_list):
        simi_matrix[idx_pair[0],idx_pair[1]] = simi_list[count]
        simi_matrix[idx_pair[1],idx_pair[0]] = simi_list[count]
    #------------------

    print('adding edges...')
    true_edge_list = []
    for i in range(len(idx_list)):
        if edge_list[i]==0:
            true_edge_list.append(idx_list[i])
    g.add_edges_from(true_edge_list)

    print('Clustering...')
    partition = community.best_partition(g)

    # decode to get label_list
    print('decoding to get label_list...')
    label_list = [0]*len(dataset)
    for key in partition:
        label_list[key] = partition[key]

    #------------------
    print('solving isolation...')
    cluster_datanum_dict = {}
    for reltype in label_list:
        if reltype in cluster_datanum_dict.keys():
            cluster_datanum_dict[reltype] += 1
        else:
            cluster_datanum_dict[reltype] = 1


    iso_reltype_list = []
    for reltype in cluster_datanum_dict:
        if cluster_datanum_dict[reltype]<=iso_thres:
            iso_reltype_list.append(reltype)

    for point_idx, reltype in enumerate(label_list):
        if reltype in iso_reltype_list:
            search_idx_list = np.argsort(simi_matrix[point_idx]) # from small to big
            for idx in search_idx_list:
                if label_list[idx] not in iso_reltype_list:
                    label_list[point_idx] = label_list[idx]
                    break
    #------------------

    return label_list, create_msg(label_list)


def Louvain_no_isolation_para(dataset, edge_measure, datatype=np.int32, para=80000):

    print('initializing the graph...')
    g = nx.Graph()
    g.add_nodes_from(np.arange(len(dataset)).tolist())
    print('preparing idx_list...')
    idx_list = []
    for i in range(len(dataset)):
        for j in range(len(dataset)):
            if j == i:
                break
            idx_list.append((i,j))

    print('calculating edges...')
    batch_count = 0
    batch_size = 10000
    left_data = np.zeros(list((batch_size,)+ dataset[0].shape), dtype=datatype)
    right_data = np.zeros(list((batch_size,)+ dataset[0].shape), dtype=datatype)
    edge_list = []
    for count,idx_pair in enumerate(idx_list):
        left_data[batch_count]=dataset[idx_pair[0]]
        right_data[batch_count]=dataset[idx_pair[1]]
        batch_count += 1
        if batch_count == batch_size:
            print('predicting...',str(round(count/len(idx_list)*100,2))+'%')
            temp_edge_list = edge_measure(left_data,right_data)
            edge_list = edge_list + temp_edge_list.reshape(batch_size).tolist()
            batch_count = 0
    if batch_count !=0:
        print('predicting...')
        temp_edge_list = edge_measure(left_data[:batch_count],right_data[:batch_count])
        edge_list = edge_list + temp_edge_list.reshape(batch_count).tolist()
    simi_list = edge_list
    edge_idx = np.argsort(edge_list)[:para]
    for idx,item in enumerate(edge_list):
        if idx in edge_idx:
            edge_list[idx] = 0 # o for edge existing
        else:
            edge_list[idx] = 1

    #------------------
    print('forming simi_matrix...')
    simi_matrix = np.zeros([len(dataset),len(dataset)])
    for count,idx_pair in enumerate(idx_list):
        simi_matrix[idx_pair[0],idx_pair[1]] = simi_list[count]
        simi_matrix[idx_pair[1],idx_pair[0]] = simi_list[count]
    #------------------

    print('adding edges...')
    true_edge_list = []
    for i in range(len(idx_list)):
        if edge_list[i]==0:
            true_edge_list.append(idx_list[i])
    g.add_edges_from(true_edge_list)

    print('Clustering...')
    partition = community.best_partition(g)

    # decode to get label_list
    print('decoding to get label_list...')
    label_list = [0]*len(dataset)
    for key in partition:
        label_list[key] = partition[key]

    #------------------
    print('solving isolation...')
    cluster_datanum_dict = {}
    for reltype in label_list:
        if reltype in cluster_datanum_dict.keys():
            cluster_datanum_dict[reltype] += 1
        else:
            cluster_datanum_dict[reltype] = 1


    iso_reltype_list = []
    for reltype in cluster_datanum_dict:
        if cluster_datanum_dict[reltype]<=5:
            iso_reltype_list.append(reltype)

    for point_idx, reltype in enumerate(label_list):
        if reltype in iso_reltype_list:
            search_idx_list = np.argsort(simi_matrix[point_idx]) # from small to big
            for idx in search_idx_list:
                if label_list[idx] not in iso_reltype_list:
                    label_list[point_idx] = label_list[idx]
                    break
    #------------------

    return label_list, create_msg(label_list)

def create_msg(label_list):
    print('creating cluster messages...')
    msg={}
    msg['num_of_nodes']=len(label_list)
    msg['num_of_clusters']=len(list(set(label_list)))
    msg['num_of_data_in_clusters']={}
    for reltype in label_list:
        try:
            msg['num_of_data_in_clusters'][reltype] += 1
        except:
            msg['num_of_data_in_clusters'][reltype] = 1

    return msg