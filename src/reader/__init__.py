#coding=GBK

import pandas as pd
import os
import numpy as np
import random as rn
'''
ʱ�������ֵ䡣
key��pat_id
value��temporal_data
dim��137
'''
temporal_dataset=[]
'''
��̬����list
shape=[pat_size,30)
'''
static_dataset=[]
'''
��ǩlist
shape=[pat_size,2]
'''
label_dataset=[]
'''
��static_dataset��label_dataset�У�sample��λ��index����ʵid��Ӧ���Ǻ�temporal_dataset��Ӧ������
example��[[3547_7],
          [45678_4],
          ...
          ...
          [65485_4]]
'''
#index_id=[]
'''
#ѡ����test������label_dataset(����static_dataset)�е�λ��index
example:[[34]
         [67]
         ...
         ...
         [2]
         [278]]
'''
data_index_test=[]

'''
#ѡ����train������label_dataset(����static_dataset)�е�λ��index
example:[[34]
         [67]
         ...
         ...
         [2]
         [278]]
'''
data_index_train=[]


#train_i=0  #ȫ�ֿ��Ʊ���
#test_i=0   #ȫ�ֿ��Ʊ���

'''
��ȡ����Դ�ļ��������������ṹ������
@param base_path: Դ�ļ�����·��
@return: ʱ������,��̬����,��ǩ���ݣ�pat_id of ��̬����
example: 
         [[0.35436,-1.48459,����0,1,1,0,0,0]
          -1.2556,-0.4523,������0,0,0,1,0,1]
          ����                                   shape(ʱ������)=[sum(time_step),137+2]
          ����
          [0.23423,1.34235������1,0,0,1,0]]
          *****************************************************************************
         [[0.2456,0.24646,0,������1,0,0,1,0]
          ����                                   shape(��̬����)=[2930,30]
          ����
          [0.235,0.5645,0,1,1,1,0,0,1,0]]
          *****************************************************************************
          [[1]
           [0]
           ����                                  shape(��ǩ����)=[2930,1]
           [0]]
'''
def read_data_source(base_path):
    numerEndIndex=81    #numeric���ݽ�ֹindex,��֮ǰ��������Ҫ��������
    '''
          ���벢����ʱ������
          1.����Դ�ļ�
          2.�������Ե�����ֵ��Ϊ����
          3.��ֵ��������
    '''
    temporal_data_source=pd.read_csv(base_path+'temporal_features.csv' ,encoding='UTF-8')     #����ʱ������
    weak_positive_list=['÷��Ѫ�����쿹��ⶨ','��Һ������������','÷��Ѫ�����쿹��ⶨ(TPPA)','�Ҹα��濹ԭ','���ο���','��Ѫ���']   #���������Ե�������
    #�����Ըĳ�����
    for i in weak_positive_list:
        temporal_data_source[i][temporal_data_source[i]==2]=1 
    numer_temporal=temporal_data_source.iloc[:,2:numerEndIndex+1]
    #numer_temporal=temporal_data_source.iloc[:,2:]
    #����ʱ������
    numer_temporal_norm = (numer_temporal - numer_temporal.mean()) /  numer_temporal.std()
    temporal_data_source.iloc[:,2:numerEndIndex+1]=numer_temporal_norm
    #DataFrame��ʽתΪlist
    #temporal_data_source_list=temporal_data_source.values.tolist()
    temporal_data_source_list=temporal_data_source.iloc[:,:numerEndIndex+1].values.tolist()
    '''
          ���벢����̬����
          1.����Դ�ļ�
          2.��ֵ��������
    '''
    static_data_source=pd.read_csv(base_path+'static_features.csv',encoding='GBK' )        #���뾲̬����
    numer_static=static_data_source.loc[:,['���','Age']]   #��ߡ�Age������Ҫ����
    #numer_static=static_data_source.iloc[:,2:]
    numer_static_norm = (numer_static - numer_static.mean()) /  numer_static.std()
    #static_data_source.iloc[:,2:4]=numer_static_norm
    static_data_source.loc[:,['���','Age']]=numer_static_norm
    static_data_source_list=static_data_source.iloc[:,2:].values.tolist()
    '''
          �����ǩ
    '''
    index_ref=static_data_source.iloc[:,0].values.tolist()  
    labels_list=static_data_source.iloc[:,1].values.tolist()  #label
    
    
    return temporal_data_source_list,static_data_source_list,labels_list,index_ref


'''
��Ϊѵ�����Ͳ��Լ����Ѳ��Լ���sample index����data_index_test(global variable)��
   1.��ȡ�ṹ����������pat_idΪ����������������
   2.���ѡȡ���Լ�
@param train_count_0:��������ѵ�������� 
@param test_count_0: �����������Լ�����
@param test_count_1:�����������Ը���
@param limit_day:�ʱ�䲽���ƣ���ȡ������ʱ�䲽=math.min(limit_day,����ʱ�䲽)
@param path: ����Դ·��
@return: ���ؾ�̬������ʱ��������input_size
'''
def generate_train_test_data(test_count_0,test_count_1,limit_day,path):
    #���÷���read_data_source��ȡ��������
    temporal_data_source_list,static_data_source_list,labels_list,index_ref=read_data_source(path)
    pat_id=temporal_data_source_list[0][0]               
    pat=[]
    '''
          ����temporal_data_source_list,��ͬһpat_id������ϳ�һ������
          ͬʱ���ҵ���Ӧid������static_data_source_list��label_list�е�index������Ӧ��static_feature��label����dataset��
    '''
    day=0
    count=0
    for i in range(len(temporal_data_source_list)):
        data=temporal_data_source_list[i]
        if (pat_id==data[0]):
            day+=1  
            if day<=limit_day:   #���ʱ�䲽���ˣ��Ͳ�Ҫ�ٶ�������
                pat.append(data[2:])
        else:
            if(day<limit_day):
                count+=1
            if day>=limit_day:
                temporal_dataset.append(pat)
                key=index_ref.index(pat_id) 
                static_dataset.append(static_data_source_list[key])
                label_dataset.append([labels_list[key]])
            pat_id=data[0]
            day=0
            pat=[]  
            pat.append(data[2:])
            day+=1
        if i==(len(temporal_data_source_list)-1):
            if day>=limit_day:
                temporal_dataset.append(pat)
                key=index_ref.index(pat_id)
                static_dataset.append(static_data_source_list[key])
                label_dataset.append([labels_list[key]])
    print(count)
    print(np.shape(temporal_dataset),np.shape(static_dataset),np.shape(label_dataset))
    '''
         ȡ�����Լ�
    '''
    df=pd.DataFrame(label_dataset,columns=['A'])
    index_1=df[df['A']==1].index.values.tolist()  #���Բ����±�index
    index_0=df[df['A']==0].index.values.tolist()  #���Բ����±�index
    #print(index_0)
    #print(index_1)
    #���ȡ���Բ��Լ�ѵ�����±ꡢ���Բ��Լ��±�
    choice_1=np.arange(len(index_1))
    rn.shuffle(choice_1)
    print(np.shape(choice_1))
    choice_0=np.random.choice(len(index_0),len(index_1),replace=False)
    print(np.shape(choice_0))
    choice_test_0=choice_0[:test_count_0]
    choice_train_0=choice_0[test_count_0:]
    choice_test_1=choice_1[:test_count_1]
    choice_train_1=choice_1[test_count_1:]
    for i in choice_test_0:
        data_index_test.append(index_0[i])
    for i in choice_test_1:
        data_index_test.append(index_1[i])
    for i in choice_train_0:
        data_index_train.append(index_0[i])
    for i in choice_train_1:
        data_index_train.append(index_1[i]) 
    rn.shuffle(data_index_train)  #����train_data���±���������   
    print(np.shape(data_index_train),np.shape(data_index_test))
    train_temporal_dataset=[]
    train_static_dataset=[]
    train_label_dataset=[]
    test_temporal_dataset=[]
    test_static_dataset=[]
    test_label_dataset=[]
    for i in range(len(temporal_dataset)):
        if i in data_index_train:
            train_temporal_dataset.append(temporal_dataset[i])
            train_static_dataset.append(static_dataset[i])
            if label_dataset[i]==[1]:
                train_label_dataset.append([1,0])
            elif label_dataset[i]==[0]:
                train_label_dataset.append([0,1])
        elif i in data_index_test:
            test_temporal_dataset.append(temporal_dataset[i])
            test_static_dataset.append(static_dataset[i])
            if label_dataset[i]==[1]:
                test_label_dataset.append([1,0])
            elif label_dataset[i]==[0]:
                test_label_dataset.append([0,1])
    #print(np.shape(train_temporal_dataset),np.shape(test_temporal_dataset))
    return train_temporal_dataset,train_static_dataset,train_label_dataset,test_temporal_dataset,test_static_dataset,test_label_dataset

#generate_train_test_data(30,30,2,os.path.abspath(os.curdir)+'/../../resources/')

