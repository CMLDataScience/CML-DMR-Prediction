#!/usr/bin/env python
# coding: utf-8

# In[9]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.patches as patches


# In[10]:


# setting input directory
DATA_DIR = 'data/'
IDATA_DIR = 'intermediate_data/'
IMG_DIR = 'out/'


# In[11]:


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
time_values = np.array(IS_data.index,dtype = int)
IS_values = IS_data.values
WBC_values = WBC_data.values

# Number of Patients
Np = IS_values.shape[1]


# check response type
dict_MR4 = {0:' Non MR4.0 ',1:'MR4.0'}
dict_MR45 = {0:' Non MR4.5 ',1:'MR4.5'}
dict_CMR = {0:' Non CMR ',1:'CMR'}

dict_list = [dict_MR4,dict_MR45,dict_CMR]

MR4_values = np.array(np.sum((IS_values<=0.01),axis=0)>0,dtype=np.int32) # MR4.0
MR45_values = np.array(np.sum((IS_values<=0.0032),axis=0)>0,dtype=np.int32)# MR4.5
# CMR_values =np.array((IS_values[8,:]==0.0032) & (IS_values[7,:]==0.0032),dtype = np.int32)
CMR_values =np.array((IS_values[8,:]<=0.0032) & (IS_values[7,:]<=0.0032),dtype = np.int32)

MR_list = [MR4_values,MR45_values,CMR_values]


# In[12]:



# import learned data
df_dynamics_parameter  =  pd.read_csv(IDATA_DIR + 'learned_data.csv',index_col=0)


# estimate time-series
epsilon = 0.00000001

c_learn =  df_dynamics_parameter.loc['c'].values[0]
xs_learn = df_dynamics_parameter.loc['xs'].values
x0_learn = (120 - IS_values[0,:])/(120 +(c_learn-1)*IS_values[0,:]) * WBC_values[0,:]

ys_learn = df_dynamics_parameter.loc['ys'].values
y0_learn =  c_learn * IS_values[0,:]/(120 +(c_learn-1)*IS_values[0,:]) * WBC_values[0,:]
 
a_learn  = df_dynamics_parameter.loc['a'].values
b_learn  = df_dynamics_parameter.loc['b'].values

xt_est = xs_learn + (x0_learn - xs_learn) * np.exp(- a_learn * np.reshape(time_values,[len(time_values),1]))
yt_est = ys_learn + (y0_learn - ys_learn) * np.exp(- b_learn *np.reshape(time_values,[len(time_values),1]))

 
IS_est = np.maximum(120 * yt_est/np.maximum(c_learn * xt_est + yt_est,epsilon),0.0032)
WBC_est = xt_est + yt_est


# In[13]:



# plot estimated time-series of IS and WBC

fig_list = IS_data.keys()

for tmp in fig_list:
    i =  np.where(IS_data.keys() == tmp)[0][0]
    plt.close()
    fig = plt.figure(figsize=(1.6, 1.6)).patch.set_alpha(0)

    plt.subplot(2,1,1)
    plt.plot(time_values,IS_values[:,i],'-s',color ='black',label = 'Measured data',linewidth = 1,markersize = 3)
    plt.plot(time_values,IS_est[:,i],':^',color ='red', label = 'Estimated data',linewidth = 1,markersize = 3)

    plt.xlim([0,24])
    plt.ylim([0.002,150])

    plt.tick_params(labelsize=6)
    plt.gca().yaxis.set_label_coords(-0.19, 0.5)
    plt.ylabel('IS [%]',fontsize=6)

    plt.title('Pt. {0:02d}'.format(WBC_data.keys()[i]),fontsize = 6)
    plt.yscale('log')


    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().yaxis.set_ticks_position('left')
    plt.gca().xaxis.set_ticks_position('bottom')
    plt.xticks([])
    plt.yticks([0.1,10])
    plt.gca().set_yticklabels(['0.1','10'])
    plt.legend(fontsize=6)



    plt.subplot(2,1,2)
    plt.plot(time_values,WBC_values[:,i],'-s',color ='black',linewidth = 1,markersize = 3)
    plt.plot(time_values,WBC_est[:,i],':^',color ='red',linewidth = 1,markersize = 3)
    plt.gca().add_patch(patches.Rectangle(xy = (0, 3100), width = 24, height = 6700,angle = 0, color = "cyan", alpha = 0.3,label = 'Normal level'))
    plt.xlim([0,24])
    plt.xticks([0,3,6,9,12,15,18,21,24])
    plt.ylim([2000,100000])
    plt.gca().yaxis.set_label_coords(-0.19, 0.5)
    plt.ylabel('WBCs [num/$\mathrm{\mu}$l]',fontsize = 6)
    plt.gca().xaxis.set_label_coords(0.5, -0.4)
    plt.xlabel('Time [month]',fontsize=6)
    plt.yscale('log')

    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().yaxis.set_ticks_position('left')
    plt.gca().xaxis.set_ticks_position('bottom')
    plt.tick_params(labelsize=6)
    plt.legend(fontsize=6)

    plt.savefig(IMG_DIR +  'Estimeted_time_series'+'{0:02d}'.format(tmp) + '.svg',bbox_inches='tight', pad_inches=0)



# In[14]:


# function making xs-a plot 
def mkxFig(DMR_values,dict_DMR):
    plt.scatter(xs_learn[DMR_values==0], a_learn[DMR_values==0],s = 20,color = 'red',label = dict_DMR[0][1:])
    plt.scatter(xs_learn[DMR_values==1], a_learn[DMR_values==1],s = 20,color = 'black',label =  dict_DMR[1])

    for i in range(Np):
        plt.text(xs_learn[i], a_learn[i]+0.1, WBC_data.keys()[i],fontsize = 6)


    plt.xlim([3000,10000])
    plt.xticks([4000,6000,8000,10000])
    plt.tick_params(labelsize=6)

    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().yaxis.set_ticks_position('left')
    plt.gca().xaxis.set_ticks_position('bottom')
    plt.legend(fontsize=6,bbox_to_anchor=(0.6, 0.2), loc='upper right', borderaxespad=0)


    plt.gca().xaxis.set_label_coords(0.5, -0.15)    
    plt.xlabel('Convergence value of \n normal WBCs [num/$\mathrm{\mu}$l]',fontsize = 6)
    plt.ylabel('Recovery rate of  \n normal WBCs [1/month]',fontsize = 6)






# In[15]:


# function making ys-b plot 
def mkyFig(DMR_values,dict_DMR):
    plt.scatter(ys_learn[DMR_values==0], b_learn[DMR_values==0],s = 20,color = 'red',label = dict_DMR[0][1:])
    plt.scatter(ys_learn[DMR_values==1], b_learn[DMR_values==1],s = 20,color = 'black',label =  dict_DMR[1])

    for i in range(Np):
        plt.text(ys_learn[i], b_learn[i]+0.1, WBC_data.keys()[i],fontsize = 6)

    plt.xscale('log')
    # plt.xlim([0.1,300])
    plt.xlim([0.05,300])
    # plt.ylim([0,7.5])
    plt.ylim([0,9.0])
    plt.legend()
    plt.gca().xaxis.set_label_coords(0.5, -0.15)
    plt.xlabel('Convergence value of \n CML cells [num/$\mathrm{\mu}$l]',fontsize = 6)
    plt.ylabel('Reduction rate of \n CML cells [1/month]',fontsize = 6)

    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().yaxis.set_ticks_position('left')
    plt.gca().xaxis.set_ticks_position('bottom')
    plt.tick_params(labelsize=6)
    plt.legend(fontsize=6)





# In[16]:



for i in range(3):

    plt.close()
    plt.figure(figsize=(5.0, 2.1)).patch.set_alpha(0)
    plt.subplots_adjust(wspace=0.4, hspace=0)
    plt.subplot(1,2,1)
    mkxFig(MR_list[i],dict_list[i])
    plt.subplot(1,2,2)
    mkyFig(MR_list[i],dict_list[i])

    plt.savefig(IMG_DIR + 'parameters' + dict_list[i][1]+'.svg',bbox_inches='tight', pad_inches=0)





