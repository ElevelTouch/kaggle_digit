from sklearn.linear_model import LogisticRegression
import os
import re
import numpy as np
from functools import reduce

#construct path to data
directory_path='/Users/feng/Desktop/digit_classfication/'
names=os.listdir(directory_path)
names.pop(0)
data_paths=[directory_path+name for name in names]


##################
#Version1:withouot wash , without cross-validation
#################
#785=1 lable + 784 pixel
train=np.arange(0,785)


#0-8 as training data
print('begin train data')
for i in range(0,9):
    path_i=data_paths[i]
    with open(path_i) as f:
        lines=f.readlines()
        data_i=np.array([list(map(int,re.split(',',line.strip()))) for line in lines])
        print(data_i.shape)
        train=np.vstack((train,data_i))
        print('train data'+str(i)+' done')
train_labels=train[:,0]
train_features=train[:,1:]

print('train data done\n','begin test data')
#9 as test data
path=data_paths[9]
with open(path) as f:
    lines=f.readlines()
    test=np.array([list(map(int,re.split(',',line.strip()))) for line in lines])
test_labels=test[:,0]
test_features=test[:,1:]
print('test data done\n','begin fit model')

#fit logistic model
clf=LogisticRegression()
clf.fit(train_features,train_labels)
print('model fit done')

test_size=test.shape[0]
score=clf.score(test_features,test_labels)
print(score)
#93.47%
