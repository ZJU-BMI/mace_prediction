#coding=GBK

import pandas as pd
import os
import numpy as np
import random as rn
'''
时序特征字典。
key：pat_id
value：temporal_data
dim：137
'''
temporal_dataset=[]
'''
静态特征list
shape=[pat_size,30)
'''
static_dataset=[]
'''
标签list
shape=[pat_size,2]
'''
label_dataset=[]
'''
在static_dataset和label_dataset中，sample的位置index和真实id对应，是和temporal_dataset对应的桥梁
example：[[3547_7],
          [45678_4],
          ...
          ...
          [65485_4]]
'''
#index_id=[]
'''
#选出的test病例在label_dataset(或者static_dataset)中的位置index
example:[[34]
         [67]
         ...
         ...
         [2]
         [278]]
'''
data_index_test=[]

'''
#选出的train病例在label_dataset(或者static_dataset)中的位置index
example:[[34]
         [67]
         ...
         ...
         [2]
         [278]]
'''
data_index_train=[]


#train_i=0  #全局控制变量
#test_i=0   #全局控制变量

'''
读取数据源文件，解析、处理并结构化样本
@param base_path: 源文件绝对路径
@return: 时序数据,静态数据,标签数据，pat_id of 静态数据
example: 
         [[0.35436,-1.48459,……0,1,1,0,0,0]
          -1.2556,-0.4523,……，0,0,0,1,0,1]
          ……                                   shape(时序数据)=[sum(time_step),137+2]
          ……
          [0.23423,1.34235……，1,0,0,1,0]]
          *****************************************************************************
         [[0.2456,0.24646,0,……，1,0,0,1,0]
          ……                                   shape(静态数据)=[2930,30]
          ……
          [0.235,0.5645,0,1,1,1,0,0,1,0]]
          *****************************************************************************
          [[1]
           [0]
           ……                                  shape(标签数据)=[2930,1]
           [0]]
'''
def read_data_source(base_path):
    numerEndIndex=81    #numeric数据截止index,这之前的数据需要进行正则化
    '''
          读入并处理时序特征
          1.读入源文件
          2.将弱阳性的特征值改为阳性
          3.数值特征正则化
    '''
    temporal_data_source=pd.read_csv(base_path+'temporal_features.csv' ,encoding='UTF-8')     #读入时序数据
    weak_positive_list=['梅毒血清特异抗体测定','尿液亚硝酸盐试验','梅毒血清特异抗体测定(TPPA)','乙肝表面抗原','丙肝抗体','隐血检查']   #含有弱阳性的特征列
    #弱阳性改成阳性
    for i in weak_positive_list:
        temporal_data_source[i][temporal_data_source[i]==2]=1 
    numer_temporal=temporal_data_source.iloc[:,2:numerEndIndex+1]
    #numer_temporal=temporal_data_source.iloc[:,2:]
    #正则化时序化数据
    numer_temporal_norm = (numer_temporal - numer_temporal.mean()) /  numer_temporal.std()
    temporal_data_source.iloc[:,2:numerEndIndex+1]=numer_temporal_norm
    #DataFrame格式转为list
    #temporal_data_source_list=temporal_data_source.values.tolist()
    temporal_data_source_list=temporal_data_source.iloc[:,:numerEndIndex+1].values.tolist()
    '''
          读入并处理静态特征
          1.读入源文件
          2.数值特征正则化
    '''
    static_data_source=pd.read_csv(base_path+'static_features.csv',encoding='GBK' )        #读入静态特征
    numer_static=static_data_source.loc[:,['身高','Age']]   #身高、Age特征需要正则化
    #numer_static=static_data_source.iloc[:,2:]
    numer_static_norm = (numer_static - numer_static.mean()) /  numer_static.std()
    #static_data_source.iloc[:,2:4]=numer_static_norm
    static_data_source.loc[:,['身高','Age']]=numer_static_norm
    static_data_source_list=static_data_source.iloc[:,2:].values.tolist()
    '''
          读入标签
    '''
    index_ref=static_data_source.iloc[:,0].values.tolist()  
    labels_list=static_data_source.iloc[:,1].values.tolist()  #label
    
    
    return temporal_data_source_list,static_data_source_list,labels_list,index_ref


'''
分为训练集和测试集，把测试集的sample index放入data_index_test(global variable)中
   1.读取结构化病例，以pat_id为索引，构建样本集
   2.随机选取测试集
@param train_count_0:阴性样本训练集个数 
@param test_count_0: 阴性样本测试集个数
@param test_count_1:阳性样本测试个数
@param limit_day:最长时间步限制，获取的样本时间步=math.min(limit_day,样本时间步)
@param path: 数据源路径
@return: 返回静态特征、时序特征的input_size
'''
def generate_train_test_data(test_count_0,test_count_1,limit_day,path):
    #调用方法read_data_source获取样本病例
    temporal_data_source_list,static_data_source_list,labels_list,index_ref=read_data_source(path)
    pat_id=temporal_data_source_list[0][0]               
    pat=[]
    '''
          遍历temporal_data_source_list,将同一pat_id的行组合成一个样本
          同时，找到对应id出现在static_data_source_list和label_list中的index，将对应的static_feature和label放入dataset中
    '''
    day=0
    count=0
    for i in range(len(temporal_data_source_list)):
        data=temporal_data_source_list[i]
        if (pat_id==data[0]):
            day+=1  
            if day<=limit_day:   #如果时间步够了，就不要再读进来了
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
         取出测试集
    '''
    df=pd.DataFrame(label_dataset,columns=['A'])
    index_1=df[df['A']==1].index.values.tolist()  #阳性病例下标index
    index_0=df[df['A']==0].index.values.tolist()  #阴性病例下标index
    #print(index_0)
    #print(index_1)
    #随机取阴性测试集训练集下标、阳性测试集下标
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
    rn.shuffle(data_index_train)  #打乱train_data的下标所在序列   
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

