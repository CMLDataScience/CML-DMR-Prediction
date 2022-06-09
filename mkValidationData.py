#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


# number of validation data
# N_DMR : Number of DMR patients
# N_NDMR : Number of non DMR patients

N_DMR = 11
N_NDMR = 11
N = N_DMR + N_NDMR
# drop number of IS values
N_IS_drop = 2
# drop number of WBCs
N_WBC_drop = 1

# other setting
c = 1.25
t = np.array([0,3,6,9,12,15,18,21,24])


# In[3]:


# making parameter of validation data
a = np.random.normal(3,1,N)
a[a<1]=1
xs = np.random.normal(6000,1000,N)

b = np.r_[np.random.normal(3,1,N_DMR),np.random.normal(2,1,N_NDMR)]
b[b<0.5]=0.5

ys = np.r_[np.random.normal(-1,0.1,N_DMR),np.random.normal(0.5,0.3,N_NDMR)]
ys = 10**ys

x0 = np.random.normal(3000,500,N)
x0[x0<100]=100
y0 = 10**np.random.normal(5,0.5,N)
y0[y0<10000]=10000




# In[4]:


# Calculating IS values and WBCs
xt = xs +  (x0 - xs)*np.e**(-a*t.reshape(-1,1))
yt = ys +  (y0 - ys)*np.e**(-b*t.reshape(-1,1))
ISt = yt/(c*xt + yt)*1.2*100
ISt[ISt<0.0032] = 0.0032
WBCt = xt + yt


# In[5]:


# numpy array to pandas dataframe 
IS_data = pd.DataFrame(ISt.T,columns=t)
WBC_data = pd.DataFrame(WBCt.T,columns=t)

# Add drop
IS_drop = np.c_[t[np.random.randint(0,len(t),N_IS_drop)],np.random.randint(0,N,N_IS_drop)]
WBC_drop = np.c_[t[np.random.randint(0,len(t),N_WBC_drop)],np.random.randint(0,N,N_WBC_drop)]

for i in range(N_IS_drop):
    IS_data[IS_drop[i][0]][IS_drop[i][1]]=''

for i in range(N_WBC_drop):
    WBC_data[WBC_drop[i][0]][WBC_drop[i][1]]=''


# In[6]:


# saving
IS_data.to_csv('data/IS_data.csv')
WBC_data.to_csv('data/WBC_data.csv')


# In[7]:


# saving reference parameters of validation data
ref_para = pd.DataFrame(np.c_[x0,xs,a,y0,ys,b],columns=["x0","xs","a","y0","ys","b"])
ref_para.to_csv('intermediate_data/ref_para.csv')

