import numpy as np
from glob import glob
import torch
import pandas as pd
import random
from os.path import join, split
from tqdm import tqdm
from sklearn import metrics
from sklearn.metrics import auc
from pyod.models.auto_encoder_torch import AutoEncoder, check_array, inner_autoencoder, check_is_fitted, \
    pairwise_distances_no_broadcast

def load_data(pkl_list,label=True):
    '''
    输入pkl的列表，进行文件加载
    label=True用来加载训练集
    label=False用来加载真正的测试集，真正的测试集无标签
    '''
    X = []
    y = []
    

    for each_pkl in pkl_list:
        item = torch.load(each_pkl)
        # 此处选取的是每个滑窗的最后一条数据，仅供参考，可以选择其他的方法，比如均值或者其他处理时序数据的网络
        # 此处选取了前7个特征，可以需求选取特征数量
        X.append(item[0][:,0:7][-1])
        if label:
            y.append(int(item[1]['label'][0]))
    X = np.vstack(X)
    if label:
        y = np.vstack(y)
    return X, y

class PyODDataset(torch.utils.data.Dataset):
    """PyOD Dataset class for PyTorch Dataloader
    """

    def __init__(self, X, y=None, mean=None, std=None):
        super(PyODDataset, self).__init__()
        self.X = X
        self.mean = mean
        self.std = std

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()#将tensor类型转换列表格式
        sample = self.X[idx, :]

        # if self.mean.any():
        #     sample = (sample - self.mean) / (self.std + 1e-5)
        #torch.from_numpy()将numpy类型转换为tensor类型
        return torch.from_numpy(sample), idx

class Car_AutoEncoder(AutoEncoder):
    
    '''
    使用autoencoder 来进行模型的训练，默认采用无监督的训练方式
    '''
    
    def fit(self, X, y=None):
        # validate inputs X and y (optional)
        X = check_array(X)
        self._set_n_classes(y)

        n_samples, n_features = X.shape[0], X.shape[1] #获取样本个数和特征个数

        # 是否进行预处理操作
        if self.preprocessing:
            self.mean, self.std = np.mean(X, axis=0), np.std(X, axis=0)
            train_set = PyODDataset(X=X, mean=self.mean, std=self.std)

        else:
            train_set = PyODDataset(X=X)
        #构建数据生成器
        train_loader = torch.utils.data.DataLoader(train_set,
                                                   batch_size=self.batch_size,
                                                   shuffle=False)

        # initialize the model ,初始化模型
        #hidden_neurons列表，可选（默认值为[64， 32]）每个隐藏层的神经元数。因此，该网络的结构为[n_features，64，32，32，64，n_features]
        #hidden_activationstr，可选（默认值='relu'）用于隐藏层的激活函数。所有隐藏层都强制使用相同类型的激活
        #batch_norm布尔值，可选（默认值为 True）是否应用批量规范化。
        #dropout_rate浮点数 （0.， 1），可选（默认值 = 0.2）要跨所有层使用的分级。
        
        self.model = inner_autoencoder(
            n_features=n_features,
            hidden_neurons=self.hidden_neurons,
            dropout_rate=self.dropout_rate,
            batch_norm=self.batch_norm,
            hidden_activation=self.hidden_activation)

        #将model放入device中
        self.model = self.model.to(self.device)

        # 训练自动编码器以找到最佳编码器
        self._train_autoencoder(train_loader)

        self.model.load_state_dict(self.best_model_dict)
        self.decision_scores_ = self.decision_function(X)#获得输入样本的异常得分
        
        self._process_decision_scores()  
        return self

    def decision_function(self, X): 
        """使用拟合的检测器预测X的原始异常分数。

            输入样本的异常分数是基于不同的检测器算法。为保持一致性，离群值分配为异常分数越大的。
        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            形状的numpy数组（n_samples，n_features）训练输入样本。仅接受稀疏矩阵，如果它们由基础估计器支持。

        Returns
        -------
        anomaly_scores : numpy array of shape (n_samples,)
            形状的numpy数组（n_samples，）输入样本的异常得分。
        """
        #对估算器执行is_fitted验证。通过验证是否存在拟合属性（以下划线结尾）来检查估计量是否拟合，否则通过给定消息引发NotFittedError。此实用程序旨在由估计器本身在内部使用，通常在其自己的预测/变换方法中使用。
        check_is_fitted(self, ['model', 'best_model_dict'])
        # X = check_array(X)

        # note the shuffle may be true but should be False
        if self.preprocessing:
            dataset = PyODDataset(X=X, mean=self.mean, std=self.std)
        else:
            dataset = PyODDataset(X=X)

        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=self.batch_size,
                                                 shuffle=False) #要设置为False
        # enable the evaluation mode
        self.model.eval()

        # construct the vector for holding the reconstruction error
        outlier_scores = np.zeros([X.shape[0], ])#形状为（X.shape[0],)
        with torch.no_grad():
            for data, data_idx in dataloader:
                data_cuda = data.to(self.device).float()
                # this is the outlier score
                outlier_scores[data_idx] = pairwise_distances_no_broadcast(
                    data, self.model(data_cuda).cpu().numpy())

        return outlier_scores

# 为了便于复现这里固定了随机种子
def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  
    np.random.seed(seed)  
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
def train(X_train,X_test,hidden_neurons,learning_rate,epochs,batch_size,contamination,drop_out,hidden_activation):
    same_seeds(42)
    clf_name = 'auto_encoder'
    clf = Car_AutoEncoder(hidden_neurons=hidden_neurons,  batch_size=batch_size, epochs=epochs,learning_rate=learning_rate,
                                    dropout_rate=drop_out,contamination=contamination,hidden_activation=hidden_activation)
    clf.fit(X_train)

    y_train_pred = clf.labels_ # 返回训练数据上的分类标签 (0: 正常值, 1: 异常值) # binary labels (0: inliers, 1: outliers)
    y_train_scores = clf.decision_scores_  # 返回训练数据上的异常值 (分值越大越异常)# raw outlier scores
    y_test_pred = clf.predict(X_test) # 返回未知数据上的分类标签 (0: 正常值, 1: 异常值) # outlier labels (0 or 1)
    y_test_scores = clf.decision_function(X_test)   #返回未知数据上的异常值 (分值越大越异常) # outlier scores
    return clf, y_test_scores

def evaluate(label,score):
    fpr, tpr, thresholds = metrics.roc_curve(label, score, pos_label=1)
    AUC = auc(fpr, tpr)
    return AUC

def main():
    data_colnames = ['volt','current','soc','max_single_volt','min_single_volt','max_temp','min_temp','timestamp']
    data_path=join(split(split(__file__)[0])[0], 'static', 'train')#存放数据的路径
    pkl_files = glob(data_path+'/*.pkl')

    ind_pkl_files = []#存放标签为0的文件
    ood_pkl_files = []#存放标签为1的文件
    for each_path in tqdm(pkl_files):
        this_pkl_file = torch.load(each_path)#下载pkl文件
        if this_pkl_file[1]['label'] == '00':
            ind_pkl_files.append(each_path)
        else:
            ood_pkl_files.append(each_path)

    random.seed(0)
    #排序并打乱存放车辆序号的集合
    random.shuffle(ind_pkl_files)
    random.shuffle(ood_pkl_files)

    train_pkl_files=[]
    for i in range(len(ind_pkl_files)//4):
        train_pkl_files.append(ind_pkl_files[i])

    test_pkl_files=[]
    for j in range(len(ind_pkl_files)//4,len(ind_pkl_files)):
        test_pkl_files.append(ind_pkl_files[j])
    for item in ood_pkl_files:
        test_pkl_files.append(item)

    X_train,y_train=load_data(train_pkl_files)

    X_test,y_test=load_data(test_pkl_files)

    _mean = np.mean(X_train, axis=0)
    _std = np.std(X_train, axis=0)
    X_train = (X_train - _mean) / (_std + 1e-4)
    X_test = (X_test - _mean) / (_std + 1e-4)

    # 定义合适的参数
    hidden_neurons=[ 32,64,64, 32]
    learning_rate=0.03
    epochs=15
    batch_size=640
    contamination=0.005
    drop_out=0.2
    hidden_activation='sigmoid'

    same_seeds(42)
    clf = Car_AutoEncoder(hidden_neurons=hidden_neurons,  batch_size=batch_size, epochs=epochs,learning_rate=learning_rate,
                                        dropout_rate=drop_out,contamination=contamination,hidden_activation=hidden_activation)
    clf.fit(X_train)
    y_test_pred = clf.predict(X_test) # 返回未知数据上的分类标签 (0: 正常值, 1: 异常值) # outlier labels (0 or 1)
    y_test_scores = clf.decision_function(X_test)   #返回未知数据上的异常值 (分值越大越异常) # outlier scores

    AUC1=evaluate(y_test,y_test_scores)

    torch.save(clf,"clf1.torch")

    clf=torch.load("clf1.torch")

    data_path3=join(split(split(__file__)[0])[0], 'static', 'test_A')
    test1_files = glob(data_path3+'/*.pkl')

    X_val,_=load_data(test1_files,label=False)
    _mean = np.mean(X_val, axis=0)
    _std = np.std(X_val, axis=0)
    X_val = (X_val - _mean) / (_std + 1e-4)
    y_val_pred = clf.predict(X_val) # 返回未知数据上的分类标签 (0: 正常值, 1: 异常值) # outlier labels (0 or 1)
    y_val_scores = clf.decision_function(X_val)   #返回未知数据上的异常值 (分值越大越异常) # outlier scores

    #记录文件名和对应的异常得分
    predict_result={}
    for i in tqdm(range(len(test1_files))):
        file=test1_files[i]
        name = split(file)[1]
        predict_result[name]=y_val_scores[i]

    predict_score=pd.DataFrame(list(predict_result.items()),columns=['file_name','score'])#列名必须为这俩个

    predict_score.to_csv('submision.csv',index = False) #保存为比赛要求的csv文件

    def scatter_3d(X, y, observed_id):
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        for i in range(7):
            if i == observed_id:
                continue
            ax.scatter(X.T[i], X.T[observed_id], y, s=0.5, label=data_colnames[i])
        ax.set_xlabel('Other features')
        ax.set_ylabel(data_colnames[observed_id])
        ax.set_zlabel('Label')
        ax.legend()
        plt.show()
    scatter_3d(X_test, y_test, observed_id=2)
    pass

if __name__ == '__main__':
    main()
