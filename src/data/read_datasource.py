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
    count=0
    base_path=os.path.abspath(os.curdir)+'/../../resources/deeplearning4/'
    #数据源
    data_source=[]
    labels=[]
    #读取文件
    files=os.listdir(base_path)

    for file in files:
        data=pd.read_csv(base_path+file,encoding='gbk')
        row=data.shape[0]  #行数
        col=data.shape[1]  #列数
        if row<day:
            continue
        data_source.append(data.iloc[0:day,2:col-2].values.tolist())
        if data.iloc[row-1,col-1]=='1':
            labels.append([1,0])
            count+=1
        else:
            labels.append([0,1])
    print(np.shape(data_source))    #shape:
    print(np.shape(labels))
    print(count)
    return data_source,labels
    get_datasource(1)

get_datasource(10)
'''
生成训练数据
'''
def get_train_data(day):
    data_datasource,labels=get_datasource(day)
    col=np.shape(data_datasource)[2]
    #normalize
    data_datasource=np.reshape(data_datasource, [-1,col])
    print(np.shape(data_datasource))
    avg=np.mean(data_datasource, axis=0)
    dev=np.std(data_datasource, axis=0)
    data_datasource=(data_datasource-avg)/dev

    #随机取
    data_datasource=np.reshape(data_datasource, [-1,day,col])
    row=len(data_datasource)
    print(row)
    test_count=row//40
    test_data=[]
    test_label=[]
    train_data=[]
    train_label=[]
    choice=np.random.choice(row,test_count,replace=False)
    for i in range(len(data_datasource)):
       if i in choice:
          test_data.append(data_datasource[i].tolist())
          test_label.append(labels[i])
       else:
          train_data.append(data_datasource[i].tolist())
          train_label.append(labels[i])
    print(np.shape(train_data),np.shape(train_label),np.shape(test_data),np.shape(test_label))
    return train_data,train_label,test_data,test_label     

