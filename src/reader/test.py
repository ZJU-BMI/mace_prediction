#coding=GBK
'''
Created on 2017��3��27��

@author: Lu.Yi
'''
import numpy as np
data=np.arange(12).reshape((3,4))

data=data.tolist()
print(data)
print(data.index([4, 5, 6, 7]))