import re
import numpy as np
import csv


#generate partition of data

with open('/Users/feng/Desktop/train.csv') as f:
    lines=f.readlines()
    data=[list(map(int,re.split(',',line.strip()))) for line in lines[1:]]
    data=np.array(data)

#divide data into 10 set , for cross-validation
data_size=data.shape[0]
feature_size=data.shape[1]
index_array=np.arange(0,data_size)
sample_size=int(data_size/10)

#begin sampling
print('partition begin')
count=0
while True:
    choice_i=np.random.choice(index_array,sample_size,replace=True)
    #save choice i
    with open(('/Users/feng/Desktop/digit_classfication/partition'+str(count)+'.csv'),'w') as f:
        writer=csv.writer(f)
        partition_i=[data[i] for i in choice_i]
        writer.writerows(partition_i)
    print('partition '+str(count)+' done')
    count+=1
    #update index_array
    index_array=np.delete(index_array,choice_i,0)
    if count==10:
        break




