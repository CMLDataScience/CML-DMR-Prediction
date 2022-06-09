#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


# setting input directory
DATA_DIR = 'data/'
IDATA_DIR = 'intermediate_data/'
IMG_DIR = 'out/'



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
CMR_values =np.array((IS_values[8,:]<=0.0032) & (IS_values[7,:]<=0.0032),dtype = np.int32) # fixed. 


MR_list = [MR4_values,MR45_values,CMR_values]


# In[2]:


def mk_result_dat(DMR_values,dict_DMR):

    #IS 3 months
    IS3_score =IS_values[1,:]
    df_IS3 = pd.DataFrame({'data': IS3_score,'group': DMR_values})
    df_IS3['group'] = df_IS3['group'].replace(dict_DMR)
    df_IS3 = df_IS3.dropna(axis = 0)

    # IS 6 manths
    IS6_score =IS_values[2,:]
    df_IS6 = pd.DataFrame({'data': IS6_score,'group': DMR_values})
    df_IS6['group'] = df_IS6['group'].replace(dict_DMR)
    df_IS6 = df_IS6.dropna(axis = 0)

    # IS 12 months
    IS12_score =IS_values[4,:]
    df_IS12 = pd.DataFrame({'data': IS12_score,'group': DMR_values})
    df_IS12['group'] = df_IS12['group'].replace(dict_DMR)
    df_IS12 = df_IS12.dropna(axis = 0)



    # Initialization of dataframe
    df_TFPN  = pd.DataFrame(index = ('TP', 'TN', 'FP','FN'))

    # counting

    IS3_TP = np.sum((IS3_score<10) &(DMR_values ==1))
    IS3_TN = np.sum((IS3_score>=10) &(DMR_values ==0))
    IS3_FP = np.sum((IS3_score<10) &(DMR_values ==0))
    IS3_FN = np.sum((IS3_score>=10) &(DMR_values ==1))
    df_TFPN['ELN guideline \n as of 3 months'] = (IS3_TP, IS3_TN, IS3_FP, IS3_FN)


    IS6_TP = np.sum((IS6_score<1) &(DMR_values ==1))
    IS6_TN = np.sum((IS6_score>=1) &(DMR_values ==0))
    IS6_FP = np.sum((IS6_score<1) &(DMR_values ==0))
    IS6_FN = np.sum((IS6_score>=1) &(DMR_values ==1))
    df_TFPN['ELN guideline \n as of 6 months'] = (IS6_TP, IS6_TN, IS6_FP, IS6_FN)


    IS12_TP = np.sum((IS12_score<0.1) &(DMR_values ==1))
    IS12_TN = np.sum((IS12_score>=0.1) &(DMR_values ==0))
    IS12_FP = np.sum((IS12_score<0.1) &(DMR_values ==0))
    IS12_FN = np.sum((IS12_score>=0.1) &(DMR_values ==1))
    df_TFPN['ELN guideline \n as of 12 months'] = (IS12_TP, IS12_TN, IS12_FP, IS12_FN)

    # Import optimized parameter c
    Np =IS_values.shape[1]

    c_CrossTest = np.zeros(Np)
    for i in range(Np): 
        DMR_tmp =np.r_[DMR_values[:i],DMR_values[i+1:]]

        df_out = pd.read_csv(IDATA_DIR + 'learned_data{0:02d}.csv'.format(i),index_col=0)
        c_CrossTest[i] = df_out.loc['c'][0]
    # Calculating validation values 
    y_CT =  c_CrossTest * IS_values/(120 +(c_CrossTest-1)*IS_values) * WBC_values
    y0_CT = y_CT[0,:]
    y1_CT = y_CT[1,:]
    y2_CT = y_CT[2,:]

    ys_CT = np.zeros(y0_CT.shape)
    b_CT = np.zeros(y0_CT.shape)



    ## case 1: IS values is satisfies the DMR condition less than 6 months
    # In case1 ys<- 0, b<- enough large value
    
    if dict_DMR[1] == 'MR4.0':
        case1 = (IS_values[1,:] <= 0.01) | (IS_values[2,:] <= 0.01)
    elif dict_DMR[1] == 'MR4.5':
        case1 = (IS_values[1,:] <= 0.0032) | (IS_values[2,:] <= 0.0032)
    elif dict_DMR[1] == 'CMR':
        case1 = (IS_values[1,:] <= 0.0032) & (IS_values[2,:] <= 0.0032)
    else:
        print('error_DMR')

    ys_CT[case1] = 0
    b_CT[case1] = 100


    ##case 2 : else

    case2  = ~case1

    ys_CT[case2] = (y0_CT[case2] *  y2_CT[case2]  - y1_CT[case2]* y1_CT[case2]) / (y0_CT[case2]  +   y2_CT[case2]  - 2 *  y1_CT[case2])
    b_CT[case2] = (np.log(y0_CT[case2] - y1_CT[case2]) -  np.log(y1_CT[case2] - y2_CT[case2]))/3

    # Initializing threthold hat_ys and hat_b, 
    # For Maximize Accuracy
    ys_Accuracy_plus = np.zeros(len(c_CrossTest))
    b_Accuracy_minus = np.zeros(len(c_CrossTest))

    # For Maximize TP
    ys_maxTP_plus = np.zeros(len(c_CrossTest))
    b_maxTP_minus = np.zeros(len(c_CrossTest))

    # For Maximize TN
    ys_maxTN_plus = np.zeros(len(c_CrossTest))
    b_maxTN_minus = np.zeros(len(c_CrossTest))


    # For Maximize F1
    ys_maxF_plus = np.zeros(len(c_CrossTest))
    b_maxF_minus = np.zeros(len(c_CrossTest))


    for  i_cross in range(len(c_CrossTest)):
        # Optimize thresholds hat_b and hat_y without validation data.
        # i_crosss: validation patiants         

        # Initializing training data 
        c = c_CrossTest[i_cross]
        DMR_cross =  np.r_[DMR_values[:i_cross],DMR_values[i_cross +1:]]
        IS_cross =  np.c_[IS_values[:,:i_cross],IS_values[:,i_cross+1:]]
        WBC_cross =  np.c_[WBC_values[:,:i_cross],WBC_values[:,i_cross+1:]]
        case1_cross =  np.r_[case1[:i_cross],case1[i_cross +1:]]
        case2_cross =  np.r_[case2[:i_cross],case2[i_cross +1:]]


        # Calculating IS and WBC to y(0 months), y(3 months), and y(6 months)
        y_cross = c * IS_cross/(120 +(c-1)*IS_cross) * WBC_cross
        y0_cross = y_cross[0,:]
        y1_cross = y_cross[1,:]
        y2_cross = y_cross[2,:]

        # Calculating y(0 months), y(3 months), and y(6 months) to ys and b
        ys_cross = np.zeros(y0_cross.shape)
        b_cross = np.zeros(y0_cross.shape)
        ## case 1: IS values is satisfies the DMR condition less than 6 months
        ys_cross[case1_cross] = 0
        b_cross[case1_cross] = 100
        ##case 2 : else
        ys_cross[case2_cross] = (y0_cross[case2_cross] *  y2_cross[case2_cross]  - y1_cross[case2_cross]* y1_cross[case2_cross]) / (y0_cross[case2_cross]  +   y2_cross[case2_cross]  - 2 *  y1_cross[case2_cross])
        b_cross[case2_cross] = (np.log(y0_cross[case2_cross] - y1_cross[case2_cross]) -  np.log(y1_cross[case2_cross] - y2_cross[case2_cross]))/3


        # Optimize threshold in the following ranges:
        ys_para =  np.arange(1,20,0.1)
        b_para =  np.arange(2,0,-0.05)
        # Initializing TFPN on the previous ranges:
        tmp_TP = np.zeros([len(ys_para),len(b_para)])
        tmp_FP = np.zeros([len(ys_para),len(b_para)])
        tmp_FN = np.zeros([len(ys_para),len(b_para)])
        tmp_TN = np.zeros([len(ys_para),len(b_para)])

        # Calculaing TFPN
        for i in range(len(ys_para)):
            for j in range(len(b_para)):
                tmp_TP[i,j] = np.sum( (ys_cross[DMR_cross==1] <= ys_para[i])& (b_cross[DMR_cross==1] >= b_para[j]) )
                tmp_TN[i,j] = np.sum( (ys_cross[DMR_cross==0] > ys_para[i])| (b_cross[DMR_cross==0] < b_para[j]) )

                tmp_FP[i,j] = np.sum( (ys_cross[DMR_cross==0] <= ys_para[i])& (b_cross[DMR_cross==0] >= b_para[j]) )
                tmp_FN[i,j] = np.sum( (ys_cross[DMR_cross==1] > ys_para[i]) | (b_cross[DMR_cross==1] < b_para[j]) )

        # Calculating Accuracy, etc.
        tmp_P  = tmp_TP + tmp_FN
        tmp_N = tmp_TN + tmp_FP

        tmp_Accuracy = (tmp_TP + tmp_TN) / (tmp_P + tmp_N)
        tmp_TPrate = tmp_TP/ tmp_P
        tmp_TNrate = tmp_TN/ tmp_N
        tmp_F  = 2*tmp_TP / (2*tmp_TP + tmp_FN + tmp_FP)

        # Calculating threshold where the Accuracy is maximmum
        ys_list = ys_para[np.where(tmp_Accuracy == np.max( tmp_Accuracy ))[0]]
        b_list = b_para[np.where(tmp_Accuracy == np.max( tmp_Accuracy ))[1]]
        n_tmp = np.argmin((ys_list - ys_list.mean())**2 + (b_list - b_list.mean())**2)
        ys_Accuracy_plus[i_cross] = ys_list[n_tmp]
        b_Accuracy_minus[i_cross] = b_list[n_tmp]


        # Maximizing TP_rate. If the candidate is not unique, we maximize TN_rate on the candidate.
        max_TPrate = np.max(tmp_TPrate)
        ys_list = ys_para[np.where((tmp_TPrate == max_TPrate ) &(tmp_TNrate == np.max(tmp_TNrate[tmp_TPrate == max_TPrate])))[0]]
        b_list = b_para[np.where((tmp_TPrate == max_TPrate ) &(tmp_TNrate == np.max(tmp_TNrate[tmp_TPrate == max_TPrate])))[1]]
        n_tmp = np.argmin((ys_list - ys_list.max())**2 + (b_list - b_list.min())**2)

        ys_maxTP_plus[i_cross] = ys_list[n_tmp]
        b_maxTP_minus[i_cross] = b_list[n_tmp]


        # Maximizing TN_rate. If the candidate is not unique, we maximize TP_rate on the candidate.
        max_TNrate = np.max(tmp_TNrate)
        ys_list = ys_para[np.where((tmp_TNrate == max_TNrate ) &(tmp_TPrate == np.max(tmp_TPrate[tmp_TNrate == max_TNrate])))[0]]
        b_list = b_para[np.where((tmp_TNrate == max_TNrate ) &(tmp_TPrate == np.max(tmp_TPrate[tmp_TNrate == max_TNrate])))[1]]
        n_tmp = np.argmin((ys_list - ys_list.min())**2 + (b_list - b_list.max())**2)

        ys_maxTN_plus[i_cross] = ys_list[n_tmp]
        b_maxTN_minus[i_cross] = b_list[n_tmp]


        # Maximizing F1 
        ys_list = ys_para[np.where(tmp_F == np.max( tmp_F ))[0]]
        b_list = b_para[np.where(tmp_F == np.max( tmp_F ))[1]]
        n_tmp = np.argmin((ys_list - ys_list.mean())**2 + (b_list - b_list.mean())**2)

        ys_maxF_plus[i_cross] = ys_list[n_tmp]
        b_maxF_minus[i_cross] = b_list[n_tmp]




    # Validation check
    TP =  np.sum((ys_CT[DMR_values==1]<=ys_Accuracy_plus[DMR_values==1]) &(b_CT[DMR_values==1]>=b_Accuracy_minus[DMR_values==1]))
    TN =  np.sum((ys_CT[DMR_values==0]>ys_Accuracy_plus[DMR_values==0]) |(b_CT[DMR_values==0]<b_Accuracy_minus[DMR_values==0]))
    FP =  np.sum((ys_CT[DMR_values==0]<=ys_Accuracy_plus[DMR_values==0]) &(b_CT[DMR_values==0]>=b_Accuracy_minus[DMR_values==0]))
    FN =  np.sum((ys_CT[DMR_values==1]>ys_Accuracy_plus[DMR_values==1]) |(b_CT[DMR_values==1]<b_Accuracy_minus[DMR_values==1]))

    df_TFPN['Maximize Accuracy'] =(TP, TN, FP, FN)

    TP =  np.sum((ys_CT[DMR_values==1]<=ys_maxTP_plus[DMR_values==1]) &(b_CT[DMR_values==1]>=b_maxTP_minus[DMR_values==1]))
    FP =  np.sum((ys_CT[DMR_values==0]<=ys_maxTP_plus[DMR_values==0]) &(b_CT[DMR_values==0]>=b_maxTP_minus[DMR_values==0]))
    FN =  np.sum((ys_CT[DMR_values==1]>ys_maxTP_plus[DMR_values==1]) |(b_CT[DMR_values==1]<b_maxTP_minus[DMR_values==1]))
    TN =  np.sum((ys_CT[DMR_values==0]>ys_maxTP_plus[DMR_values==0]) |(b_CT[DMR_values==0]<b_maxTP_minus[DMR_values==0]))

    df_TFPN['Maximize Sensitivity'] =(TP, TN, FP, FN)


    TP =  np.sum((ys_CT[DMR_values==1]<=ys_maxTN_plus[DMR_values==1]) &(b_CT[DMR_values==1]>=b_maxTN_minus[DMR_values==1]))
    FP =  np.sum((ys_CT[DMR_values==0]<=ys_maxTN_plus[DMR_values==0]) &(b_CT[DMR_values==0]>=b_maxTN_minus[DMR_values==0]))
    FN =  np.sum((ys_CT[DMR_values==1]>ys_maxTN_plus[DMR_values==1]) |(b_CT[DMR_values==1]<b_maxTN_minus[DMR_values==1]))
    TN =  np.sum((ys_CT[DMR_values==0]>ys_maxTN_plus[DMR_values==0]) |(b_CT[DMR_values==0]<b_maxTN_minus[DMR_values==0]))

    df_TFPN['Maximize Specificity'] =(TP, TN, FP, FN)


    TP =  np.sum((ys_CT[DMR_values==1]<=ys_maxF_plus[DMR_values==1]) &(b_CT[DMR_values==1]>=b_maxF_minus[DMR_values==1]))
    FP =  np.sum((ys_CT[DMR_values==0]<=ys_maxF_plus[DMR_values==0]) &(b_CT[DMR_values==0]>=b_maxF_minus[DMR_values==0]))
    FN =  np.sum((ys_CT[DMR_values==1]>ys_maxF_plus[DMR_values==1]) |(b_CT[DMR_values==1]<b_maxF_minus[DMR_values==1]))
    TN =  np.sum((ys_CT[DMR_values==0]>ys_maxF_plus[DMR_values==0]) |(b_CT[DMR_values==0]<b_maxF_minus[DMR_values==0]))


    df_TFPN['Maximize F'] =(TP, TN, FP, FN)



    #  calc results
    df_TFPN.loc['P'] =  df_TFPN.loc['TP'] + df_TFPN.loc['FN']
    df_TFPN.loc['N'] =  df_TFPN.loc['TN'] + df_TFPN.loc['FP']
    df_TFPN.loc['ALL'] =  df_TFPN.loc['N'] + df_TFPN.loc['P']

    df_TFPN.loc['Accuracy'] = (df_TFPN.loc['TP'] + df_TFPN.loc['TN']) / df_TFPN.loc['ALL']
    df_TFPN.loc['Sensitivity [True Positive Rate]'] = df_TFPN.loc['TP']/ df_TFPN.loc['P']
    df_TFPN.loc['Specificity [True Negative Rate]'] = df_TFPN.loc['TN']/ df_TFPN.loc['N']

    return df_TFPN



# In[3]:


i = 0
df = mk_result_dat(MR_list[0],dict_list[0])
df.to_csv(IDATA_DIR + 'Score_MR4.0.csv')

i = 1
df = mk_result_dat(MR_list[i],dict_list[i])
df.to_csv(IDATA_DIR + 'Score_MR4.5.csv')

i = 2
df = mk_result_dat(MR_list[i],dict_list[i])
df.to_csv(IDATA_DIR + 'Score_CMR.csv')




# In[4]:


# make fig function
def mkFig(df_TFPN,title):
    df_plot = df_TFPN.loc[['Accuracy', 'Sensitivity [True Positive Rate]', 'Specificity [True Negative Rate]']]
    df_plot = df_plot[['ELN guideline \n as of 3 months', 'ELN guideline \n as of 6 months', 'ELN guideline \n as of 12 months', 'Maximize Accuracy','Maximize Sensitivity', 'Maximize Specificity']]
    df_plot = df_plot.T
    df_plot.plot.bar(figsize=(3.8,2.0),width = 0.8,color = ['magenta','cyan','lime'])


    plt.legend(fontsize=6, loc='lower right', borderaxespad=0)
    # Accuracy text
    for i in range(len(df_plot.index)):
        tmp_values = df_plot.values[i,0]
        plt.text( i-0.33,tmp_values-0.11, '{0:.02f}'.format(tmp_values),fontsize = 6,rotation=90)
        tmp_values = df_plot.values[i,1]
        plt.text( i-0.07,tmp_values-0.11, '{0:.02f}'.format(tmp_values),fontsize = 6,rotation=90)
        tmp_values = df_plot.values[i,2]
        plt.text( i+0.21,tmp_values-0.11, '{0:.02f}'.format(tmp_values),fontsize = 6,rotation=90)


    plt.xlim([-0.5,5.5])
    plt.tick_params(labelsize=6)
    plt.ylabel('Classification performance',fontsize = 6)

    plt.title(title,fontsize =8)






# In[5]:


# make figure
df = pd.read_csv(IDATA_DIR + 'Score_MR4.0.csv',index_col=0)
mkFig(df,'MR4.0')
plt.savefig(IMG_DIR + 'Score_MR4.0.svg',bbox_inches='tight', pad_inches=0)

df = pd.read_csv(IDATA_DIR + 'Score_MR4.5.csv',index_col=0)
mkFig(df,'MR4.5')
plt.savefig(IMG_DIR + 'Score_MR4.5.svg',bbox_inches='tight', pad_inches=0)

df = pd.read_csv(IDATA_DIR + 'Score_CMR.csv',index_col=0)
mkFig(df,'CMR')
plt.savefig(IMG_DIR + 'Score_CMR.svg',bbox_inches='tight', pad_inches=0)





