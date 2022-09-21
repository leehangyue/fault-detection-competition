# -*- coding: utf-8 -*-
"""
Created on Sat Sep 17 13:54:37 2022

@author: zzy
"""

from os.path import join, split
import numpy as np
import random
import torch
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras
from glob import glob
from sklearn import metrics
from sklearn.metrics import auc

def load_data(pkl_list,label=True):
    '''
    输入pkl的列表，进行文件加载
    label=True用来加载训练集
    label=False用来加载真正的测试集，真正的测试集无标签
    '''
    X = []
    y = []
    
    for  each_pkl in pkl_list:
        item = torch.load(each_pkl)
        # LSTM读取所有数据，但不考虑时间戳间隔不一致的问题
        # 形成的X array形状变为 n×256×7
        X.append(item[0][:,0:7])
        if label:
            y.append(int(item[1]['label'][0]))
    X = np.array(X)
    if label:
        y = np.vstack(y)
    return X, y


def BuildLSTM(TimeStep, InputColNum, OutStep, MiddleNeurons, HiddenActi = 'tanh', LearnRate = 1e-4):
    """
    Parameters
    ----------
    TimeStep : int
        每个sample的序列长度，这个case里是256
    InputColumnNum : int
        特征数，目前是7
    OutStep : int
        输出序列长度，这里分类预测长度为1
    MiddleNeurons : list
        LSTM及全连接层神经元数
    HiddenActi : str
        tf支持的激活函数
    LearnRate : float
        keras.optimizers.Adam学习率

    Returns
    -------
    keras.models.Models的Model类对象
    """
    
    #输入层
    InputLayer = keras.layers.Input(shape = (TimeStep, InputColNum))
    
    #中间层，一层LSTM加若干全连接层
    Middle = []
    Middle.append(keras.layers.LSTM(MiddleNeurons[0], activation = HiddenActi)(InputLayer))
    for NeuronNum in MiddleNeurons[1:]:
        Middle.append(keras.layers.Dense(NeuronNum, activation = HiddenActi)(Middle[-1]))
    
    #全连接输出层
    #???输出层使用softmax层会与交叉熵loss函数重复，所以只使用一般的全连接层？
    OutputLayer = keras.layers.Dense(OutStep, activation = 'softsign')(Middle[-1])
    
    Model = keras.models.Model(inputs = InputLayer, outputs = OutputLayer)
    optimizer = keras.optimizers.Adam(learning_rate = LearnRate)
    #from_logits = False对应经过softmax函数的输出
    Model.compile(optimizer = optimizer, loss = keras.losses.BinaryCrossentropy(from_logits = False),
                  metrics = ['binary_accuracy'])
    Model.summary()
    return Model
    
def evaluate(label,score):
    fpr, tpr, thresholds = metrics.roc_curve(label, score, pos_label=1)
    AUC = auc(fpr, tpr)
    return AUC   
    

if __name__ == '__main__':
    data_path = join(split(split(__file__)[0])[0], 'static', 'train')
    pkl_files = glob(data_path+'/*.pkl')
    
    ind_pkl_files = []#存放标签为0的文件
    ood_pkl_files = []#存放标签为1的文件
    for each_path in tqdm(pkl_files):
        this_pkl_file = torch.load(each_path)#读取pkl文件
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
    
    # X形状改变后，去中心化操作相应改变
    _mean = np.mean(X_train, axis=(0,1))
    _std = np.std(X_train, axis=(0,1))
    X_train = (X_train - _mean) / (_std + 1e-4) / 6
    X_test = (X_test - _mean) / (_std + 1e-4) / 6
    
    MiddleNeurons = [40, 20, 20]
    HiddenActi = 'tanh'
    epochs = 20
    BatchSize = 32768
    
    LSTMModel = BuildLSTM(256, 7, 1, MiddleNeurons, HiddenActi = HiddenActi, LearnRate=9e-4)
    EarlyStop = keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 10)
    
    # log路径打开cmd，运行tensorboard --logdir = train，浏览器访问localhost:6006
    LogDir = join(split(__file__)[0], 'LSTMlog')
    TensorboardCallback = tf.keras.callbacks.TensorBoard(log_dir = LogDir, histogram_freq = 10)
    
    history = LSTMModel.fit(X_train, y_train, epochs = epochs, batch_size = BatchSize, validation_split = 0.2, verbose = 1,
                            callbacks = [TensorboardCallback, EarlyStop])
    
    yTrainPred = LSTMModel.predict(X_train)
    yTestPred = LSTMModel.predict(X_test)
    
    AUC1 = evaluate(y_test, yTestPred)
    print('AUC1  =', AUC1)
    print('yTrainPred minmax = ', np.min(yTrainPred), ', ', np.max(yTrainPred))
    print('yTestPred minmax = ', np.min(yTestPred), ', ', np.max(yTestPred))
    
    LSTMModel.save(join(split(__file__)[0], 'LSTMModel'))
