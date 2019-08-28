import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

colValue = ['black', 'r', 'olive','blue','sandybrown',  'bisque', 
'gold','yellow','greenyellow','turquoise','cyan','dodgerblue','darkviolet',
'fuchsia','deeppink','moccasin' , 'brown', 'lightcoral']
markValue = ['o', '>','<', 'v', '+' , '8', 's', 'p', '*', 'h', 'x', '^', '<', 'd']
# markValue = ['o']
colValue_NA = 'lightgrey'
markValue_NA = '*'
def plot(data_as_array, label_list, save_figure_file = None, method='pca', NA=True, best=4):
    '''for at most 8 classes'''
    if method=='pca':
        pca = PCA(n_components=2)
        pca.fit(data_as_array)
        result = pca.transform(data_as_array)
    elif method =='tsne':
        tsne = TSNE(n_components=2, init='pca', random_state=None, perplexity=20)
        result = tsne.fit_transform(data_as_array)

    if save_figure_file is not None:
        plt.switch_backend('agg')
    fig = plt.figure(1)

    # consider NA
    if NA:
        label_of_cluster = []
        num_of_cluster = []
        for item in label_list:
            if item in label_of_cluster:
                num_of_cluster[label_of_cluster.index(item)]+=1
            else:
                label_of_cluster.append(item)
                num_of_cluster.append(1)

        large_clusters = np.array(num_of_cluster).argsort()[-best:]

        for i,item in enumerate(label_list):
            if label_of_cluster.index(item) not in large_clusters:
                label_list[i] = 'Other'

    cluster_list = set(label_list)
    other_exist = 0
    if 'Other' in cluster_list:
        other_exist = 1
        cluster_list.remove('Other')
    cluster_list = list(cluster_list)

    class_num = len(cluster_list)
    coldict = {}
    markdict = {}
    for i,item in enumerate(cluster_list):
        coldict[item] = colValue[i % len(colValue)]
        markdict[item] = markValue[i % len(markValue)]

    for i, item in enumerate(cluster_list):
        temp_id = [i for i,label in enumerate(label_list) if label==item]
        plt.scatter(result[temp_id,0],result[temp_id,1],c=coldict[item],marker=markdict[item],label=str(i),s=8)
    if other_exist:
        temp_id = [i for i,label in enumerate(label_list) if label=='Other']
        plt.scatter(result[temp_id,0],result[temp_id,1],c=colValue_NA,marker=markValue_NA,label='Other',s=8)

    plt.xticks([])
    plt.yticks([])
    # plt.legend()
    if save_figure_file is None:
        plt.show()
    else:
        fig.savefig(save_figure_file)
    return

def plot2(data_as_array, true_label_list, predicted_label_list, save_figure_file = None, method='pca'):
    '''for at most 8 classes'''

    if method=='pca':
        pca = PCA(n_components=2)
        pca.fit(data_as_array)
        result = pca.transform(data_as_array)
    elif method =='tsne':
        tsne = TSNE(n_components=2, init='pca', random_state=0)
        result = tsne.fit_transform(data_as_array)

    if save_figure_file is not None:
        plt.switch_backend('agg')
    fig = plt.figure(1)

    plt.subplot(121)
    class_num = len(list(set(true_label_list)))
    if class_num > len(colValue):
        print('too many true classes! Different classes may be mixed in the result figure!')
    coldict = {}
    markdict = {}
    for i,item in enumerate(list(set(true_label_list))):
        coldict[item] = colValue[i % len(colValue)]
        markdict[item] = markValue[i % len(markValue)]

    for i, item in enumerate(list(set(true_label_list))):
        temp_id = [i for i,label in enumerate(true_label_list) if label==item]
        plt.scatter(result[temp_id,0],result[temp_id,1],c=coldict[item],marker=markdict[item],label=str(item),s=8)
    plt.legend()
    plt.title('true_label')

    plt.subplot(122)
    class_num = len(list(set(predicted_label_list)))
    if class_num > len(colValue):
        print('too many predicted classes! Different classes may be mixed in the result figure!')
    coldict = {}
    markdict = {}
    for i,item in enumerate(list(set(predicted_label_list))):
        coldict[item] = colValue[i % len(colValue)]
        markdict[item] = markValue[i % len(markValue)]

    for i, item in enumerate(list(set(predicted_label_list))):
        temp_id = [i for i,label in enumerate(predicted_label_list) if label==item]
        plt.scatter(result[temp_id,0],result[temp_id,1],c=coldict[item],marker=markdict[item],label=str(item),s=8)
    plt.legend()
    plt.title('predicted_label')

    if save_figure_file is None:
        plt.show()
    else:
        fig.savefig(save_figure_file)
    return



if __name__=='__main__':
    ''' for test '''
    data_as_array = np.array([[1,2,3,4,5],[5,4,3,2,1],[3,3,3,3,6],[4,2,5,8,3]])
    true_label_list = [0,1,0,2]
    predicted_label_list = [3,3,4,4]
    plot2(data_as_array,true_label_list,predicted_label_list,save_figure_file='test.png')