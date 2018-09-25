#coding=gbk
'''
Created on 2017年3月21日

@author: Lu.Yi
'''


import pandas as pd
import os
import numpy as np


'''
获取data source
'''
def get_datasource(day):
    base_path=os.path.abspath(os.curdir)+'/../../resources/'
    #获取static_feature
    df=pd.read_csv(base_path+'static_deeplearning.csv',encoding='GBK')
    ids=df['pat_id'].values.tolist()
    static_data=df.iloc[:,1:].values.tolist()
    #数据源
    temporal_data_source=[]
    static_data_source=[]
    labels=[]
    #读取文件
    files=os.listdir(base_path+'deeplearning4/')

    for file in files:
        pat_id=os.path.basename(file).split('N')[0]
        index=ids.index(pat_id)
        static_data_source.append(static_data[index])
        data=pd.read_csv(base_path+'deeplearning4/'+file,encoding='gbk')
        row=data.shape[0]  #行数
        col=data.shape[1]  #列数
        if row<day:
            continue
        temporal_data_source.append(data.iloc[0:day,2:col-2].values.tolist())
        if data.iloc[row-1,col-1]=='1':
            labels.append([1,0])
            #count+=1
        else:
            labels.append([0,1])
    print(np.shape(temporal_data_source))    #shape:
    print(np.shape(labels))
    #print(count)
    return static_data_source,temporal_data_source,labels

'''
生成训练数据
'''
def get_train_data(day):
    static_data_source,temporal_data_source,labels=get_datasource(day)
    col=np.shape(temporal_data_source)[2]
    #normalize
    temporal_data_source=np.reshape(temporal_data_source, [-1,col])
    print(np.shape(temporal_data_source))
    avg=np.mean(temporal_data_source, axis=0)
    dev=np.std(temporal_data_source, axis=0)
    temporal_data_source=(temporal_data_source-avg)/dev

    #随机取
    data_datasource=np.reshape(temporal_data_source, [-1,day,col])
    row=len(data_datasource)
    print(row)
    test_count=row//30
    test_temporal_data=[]
    test_static_data=[]
    test_label=[]
    train_temporal_data=[]
    train_static_data=[]
    train_label=[]
    choice=np.random.choice(row,test_count,replace=False)
    for i in range(len(data_datasource)):
       if i in choice:
          test_temporal_data.append(data_datasource[i].tolist())
          test_label.append(labels[i])
       else:
          train_temporal_data.append(data_datasource[i].tolist())
          train_label.append(labels[i])
    print(np.shape(train_temporal_data),np.shape(train_label),np.shape(test_temporal_data),np.shape(test_label))
    return static_data_source,train_temporal_data,train_label,test_temporal_data,test_label     

