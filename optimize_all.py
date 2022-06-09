#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
# import tensorflow  as tf
import tensorflow.compat.v1 as tf # for TensorFlow 2
tf.disable_v2_behavior()
import time


# setting input directory
DATA_DIR = 'data/'



# open data
IS_data =  pd.read_csv(DATA_DIR + 'IS_data.csv',index_col=0).T
WBC_data =  pd.read_csv(DATA_DIR + 'WBC_data.csv',index_col=0).T

# drop nan data
IS_data = IS_data.dropna(axis=1)
WBC_data = WBC_data.dropna(axis=1)
keys = IS_data.keys() & WBC_data.keys()

IS_data = IS_data[keys]
WBC_data = WBC_data[keys]


# object to math type

IS_data.index =  IS_data.index.map(int)
WBC_data.index =  WBC_data.index.map(int)


# To numpy data
IS_values = IS_data.values
WBC_values = WBC_data.values

# Number of Patients
Np = IS_values.shape[1]


# Definition of MR4.0, DMR, CMR
# 1: Cure, 0: Non cure 

CMR_values =np.array((IS_values[8,:]<=0.0032) & (IS_values[7,:]<=0.0032),dtype = np.int32)
DMR_values =np.array(IS_values[8,:]<=0.0032,dtype = np.int32)
MR4_values =np.array(IS_values[4,:] <= 0.01,dtype = np.int32)


# Optimization 

# Learning parameters:
# scalar value : c
# vector value : a, b, xs, ys
# 
# explanatory value : IS(0M), WBC(0M)
# objective value : \sum_{t = 3M, ... , 24M} log(IS(t))
# estimation func :  mean squre error of  log(IS(t))

# Setting of parameters
xs_min = 3100
xs_max = 9800
epsilon = 0.00000001



# Designing models

# change time scaling from [3 months, 6 months, ...] to [1,2,...]
times = tf.reshape(tf.constant([1,2,3,4,5,6,7,8],tf.float32),[8,1])

# Initialization of parameters
xs_tmp  = tf.Variable(np.ones(Np),dtype=tf.float32)
xs =  6000 * xs_tmp
a = tf.Variable(0.5 * np.ones(Np),dtype=tf.float32)

ys_tmp  = tf.Variable(np.ones(Np),dtype=tf.float32)
ys =  10 * ys_tmp
b  = tf.Variable( 5*np.ones(Np),dtype=tf.float32)

c =  tf.Variable(1.0, dtype=tf.float32)

# Initialization of placeholders
_IS = tf.placeholder(tf.float32, shape = [9,Np])
_WBC = tf.placeholder(tf.float32, shape = [9,Np])

# Calculating x0 and y0 from IS(0) and WBC(0)
x0 =  (120- _IS [0,:]) / (120 + (c - 1) *_IS [0,:])* _WBC[0,:]
y0 =  c* _IS[0,: ] / (120 + (c - 1)*_IS [0,:])* _WBC[0,:]

# Calculating x(t) and y(t) from times, x0 and y0
xt  =  xs +  (x0  -  xs) * tf.exp(-a* times)
yt  =  ys +  (y0  -  ys) * tf.exp(-b* times)

# Observation of IS(t) and WBC(t). 
# 0.0032 is observation limit of IS values.
IS =  tf.maximum(yt / (c * xt + yt) * 120, 0.0032)
# If WBC == 0, np.log(WBC) == inf in loss function,
WBC =  tf.maximum(xt + yt,epsilon)

# ys minimal values estimated by the observation limit of IS.
ys_min = xs *0.0032/120*c

# Constraint conditions
# Condition 1 : upper and lower bound of normal WBCs
sc1 = tf.reduce_sum(tf.nn.relu( - xs + xs_min))  + tf.reduce_sum(tf.nn.relu( xs - xs_max))

# Condition 2 : parameters a and b is positive
sc2 = tf.reduce_sum(tf.nn.relu( - a)) + tf.reduce_sum(tf.nn.relu( - b)) 

# Condition 3 : parameters ys is larger than ys_min
sc3 = tf.reduce_sum(tf.nn.relu( - ys+ys_min))



# Designing loss function
loss  = tf.reduce_mean(tf.square( tf.log(IS) - tf.log(_IS[1:,:]))) + tf.reduce_mean(tf.square( tf.log(WBC) - tf.log(_WBC[1:,:]))) + 100 * (sc1 +  sc2 + sc3)
# Setting optimizer
train =  tf.train.AdamOptimizer(0.00001).minimize(loss)


# Optimization using all partients data

# caluclation starting
start_time = time.time()

# Iteration 
## if you want to finish the optimization quickly, you can choose the lower  iteration number N.
N = 10000000
# N = 10000


# Initialization
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Setting learning data
IS_learn  = IS_values
WBC_learn = WBC_values
keys_learn = IS_data.keys()
# learning
for  i in range(N):
    sess.run(train,feed_dict={_IS:IS_learn,_WBC:WBC_learn})
    if (i%(N//100) == 0):
        print('{0:02d} %'.format(i//(N//100)))
loss_data = sess.run(loss, feed_dict={_IS:IS_learn,_WBC:WBC_learn})
# get parameters(time re-scaling )
c_learn = sess.run(c) 
a_learn = sess.run(a) / 3
xs_learn = sess.run(xs)
b_learn =  sess.run(b) / 3
ys_learn =  sess.run(ys)

x0_learn =  sess.run(x0, feed_dict={_IS:IS_learn,_WBC:WBC_learn})
y0_learn =  sess.run(y0, feed_dict={_IS:IS_learn,_WBC:WBC_learn})

# save data
learned_values = np.vstack((xs_learn,ys_learn,a_learn,b_learn,c_learn* np.ones(xs_learn.shape),loss_data* np.ones(xs_learn.shape)))
learned_data  =  pd.DataFrame(learned_values,index =['xs','ys','a','b','c','loss_data'], columns=keys_learn )
learned_data.to_csv('intermediate_data/learned_data.csv')


# Calculation finishing
elapsed_time = time.time() - start_time
print('elapsed_time ={0:.04f}'.format(elapsed_time))



