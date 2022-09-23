# -*- coding: utf-8 -*-
"""
Created on Sat Sep 17 13:54:37 2022

@author: zzy
"""

from os.path import join, split
import numpy as np
import pandas as pd
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
    
    for each_pkl in tqdm(pkl_list):
        item = torch.load(each_pkl)
        # LSTM读取所有数据，但不考虑时间戳间隔不一致的问题
        # 形成的X array形状变为 n×(256×8)
        item_x = np.array(item[0][:,0:7])
        n_timestep =item_x.shape[0]
        item_x = np.hstack([item_x, item[1]['mileage'] * np.ones([n_timestep, 1])])
        X.append(item_x)
        if label:
            y.append(int(item[1]['label'][0]))
    X = np.array(X)
    if label:
        y = np.vstack(y)
    return X, y


def BuildLSTM(TimeStep, InputColNum, OutStep, MiddleNeurons, optimizer, HiddenActi = 'tanh'):
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
    shrink_ratio = TimeStep // MiddleNeurons[0]
    Middle.append(keras.layers.AveragePooling1D(pool_size=shrink_ratio, strides=shrink_ratio, padding="valid")(InputLayer))
    Middle.append(keras.layers.SeparableConv1D(filters=4, kernel_size=MiddleNeurons[0], strides=MiddleNeurons[0], padding="valid", activation=HiddenActi)(Middle[-1]))
    # Middle.append(keras.layers.LSTM(MiddleNeurons[0], activation = HiddenActi)(Middle[-1]))
    Middle.append(keras.layers.Flatten()(Middle[-1]))
    for NeuronNum in MiddleNeurons[1:]:
        Middle.append(keras.layers.Dense(NeuronNum, activation = HiddenActi)(Middle[-1]))
    
    #全连接输出层
    OutputLayer = keras.layers.Dense(OutStep, activation = 'sigmoid')(Middle[-1])
    
    Model = keras.models.Model(inputs = InputLayer, outputs = OutputLayer)
    #from_logits = False对应经过softmax函数的输出
    Model.compile(optimizer = optimizer, loss = keras.losses.BinaryCrossentropy(from_logits = False),
                  metrics = ['binary_accuracy'])
    Model.summary()
    return Model
    
def evaluate(label,score):
    fpr, tpr, thresholds = metrics.roc_curve(label, score, pos_label=1)
    AUC = auc(fpr, tpr)
    # AUC2 = 0
    # pos_id = np.argwhere(label > 0.5).T[0]
    # neg_id = np.argwhere(label < 0.5).T[0]
    # pos_score = score[pos_id]
    # neg_score = score[neg_id]
    # for ps in pos_score:
    #     AUC2 += np.sum((neg_score < ps) * 1.) + np.sum((neg_score == ps) * 0.5)
    # AUC2 = AUC2 / (len(pos_score) * len(neg_score))
    return AUC   
    

if __name__ == '__main__':
    data_path = join(split(split(__file__)[0])[0], 'static', 'train')
    # data_path = './train'
    pkl_files = glob(data_path+'/*.pkl')
    
    random.seed(1)
    #排序并打乱存放车辆序号的集合
    random.shuffle(pkl_files)
    
    pos_pkl_files = []
    neg_pkl_files = []
    for f in pkl_files:
        pkl_data = torch.load(f)
        if int(pkl_data[1]['label'][0]) == 1:
            pos_pkl_files.append(f)
        else:
            neg_pkl_files.append(f)
    n_pos = len(pos_pkl_files)
    n_neg = len(neg_pkl_files)
    
    train_pkl_files=[]
    test_pkl_files=[]
    for i in range(n_pos):
        if i < n_pos//2:
            train_pkl_files.append(pos_pkl_files[i])
        else:
            test_pkl_files.append(pos_pkl_files[i])
    for i in range(n_neg):
        if i < 5*n_pos//2:
            train_pkl_files.append(neg_pkl_files[i])
        else:
            test_pkl_files.append(neg_pkl_files[i])
    
    random.seed(0)
    #排序并打乱存放车辆序号的集合
    random.shuffle(train_pkl_files)

    X_train,y_train=load_data(train_pkl_files)
    X_test,y_test=load_data(test_pkl_files)
    
    # X形状改变后，去中心化操作相应改变
    _mean = np.mean(X_train, axis=(0,1))
    _std = np.std(X_train, axis=(0,1))
    X_train = (X_train - _mean) / (_std + 1e-4)
    X_test = (X_test - _mean) / (_std + 1e-4)
    
    MiddleNeurons = [4, 4]
    HiddenActi = 'tanh'
    epochs = 100
    BatchSize = 64
    # optimizer = keras.optimizers.RMSprop(lr=1e-3, rho=0.9, epsilon=1e-08, decay=0.0)
    optimizer = keras.optimizers.Adam(learning_rate = 5e-4)
    
    LSTMModel = BuildLSTM(256, 8, 1, MiddleNeurons, HiddenActi = HiddenActi, optimizer=optimizer)
    
    # 载入模型
    # LSTMModel = keras.models.load_model(join(split(__file__)[0], 'LSTMModel'))
    
    EarlyStop = keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 15)
    
    # log路径打开cmd，运行tensorboard --logdir = train，浏览器访问localhost:6006
    # LogDir = 'LSTMlog'
    LogDir = join(split(__file__)[0], 'LSTMlog')
    TensorboardCallback = tf.keras.callbacks.TensorBoard(log_dir = LogDir, histogram_freq = 10)

    # def adapt_learning_rate(epoch):
    #     return 3e-3 * (10 / (10 + max(0, epoch - 20)))
    # my_lr_scheduler = keras.callbacks.LearningRateScheduler(adapt_learning_rate)
    # history = LSTMModel.fit(X_train, y_train, epochs = epochs, batch_size = BatchSize, validation_split = 0.2, verbose = 1,
    #                         callbacks = [TensorboardCallback, EarlyStop, my_lr_scheduler], shuffle=False)
    for epoch in range(epochs):  # 手动shuffle batch样本，相当于固定fit中shuffle的种子，慢但可重复
        indices = np.arange(X_train.shape[0])
        random.seed(epoch+2077)
        random.shuffle(indices)
        print('Epoch %d/%d'%(epoch+1,epochs))
        def adapt_learning_rate(ignored_epoch):
            return 3e-3 * (10 / (10 + max(0, epoch - 50)))
        my_lr_scheduler = keras.callbacks.LearningRateScheduler(adapt_learning_rate)
        history = LSTMModel.fit(X_train[indices], y_train[indices], epochs=1, batch_size=BatchSize, validation_split=0.2, verbose=1,
                                callbacks=[TensorboardCallback, EarlyStop, my_lr_scheduler], shuffle=False)

    # yTrainPred = LSTMModel.predict(X_train)
    yTestPred = LSTMModel.predict(X_test)
    
    AUC1 = evaluate(y_test, yTestPred)
    print('AUC1  =', AUC1)
    # print('yTrainPred minmax = ', np.min(yTrainPred), ', ', np.max(yTrainPred))
    print('yTestPred minmax = ', np.min(yTestPred), ', ', np.max(yTestPred))
    pos_score = yTestPred[np.argwhere(y_test >= 0.5).T[0], 0]
    neg_score = yTestPred[np.argwhere(y_test < 0.5).T[0], 0]
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2)
    axs[0].hist(neg_score, bins=50, log=True, color='b', label='Score dist with label 0')
    axs[1].hist(pos_score, bins=50, log=True, color='r', label='Score dist with label 1')
    axs[0].legend()
    axs[1].legend()
    plt.show()
    
    LSTMModel.save(join(split(__file__)[0], 'LSTMModel'), 'LSTMModel')
    # LSTMModel.save('./LSTMModel', 'LSTMModel')
    
    testA_path = join(split(split(__file__)[0])[0], 'static', 'test_A')
    # testA_path = './test_A'
    testA_pkl_files = glob(testA_path+'/*.pkl')
    X_testA, y_testA = load_data(testA_pkl_files, label = False)
    X_testA = (X_testA - _mean) / (_std + 1e-4)
    yTestAPred = LSTMModel.predict(X_testA)
    predict_score = pd.DataFrame([split(fname)[1] for fname in testA_pkl_files])
    predict_score = pd.concat([predict_score, pd.DataFrame(yTestAPred)], axis = 1)
    predict_score.columns = ['file_name', 'score']
    predict_score = predict_score.sort_values(by='file_name')
    # predict_score.to_csv('submission.csv')
    predict_score.to_csv(join(split(__file__)[0], 'submission.csv'), index=False)
