#coding=GBK
'''
Created on 2017年3月27日

@author: Lu.Yi
'''
import numpy as np
data=np.arange(12).reshape((3,4))

data=data.tolist()
print(data)
print(data.index([4, 5, 6, 7]))