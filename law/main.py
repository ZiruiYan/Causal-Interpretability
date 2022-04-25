#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 23:43:07 2022

@author: rain
"""
import os
import tempfile
import pandas as pd
import six.moves.urllib as urllib
import pprint

# import tensorflow_model_analysis as tfma
# from google.protobuf import text_format

# import tensorflow as tf
# tf.compat.v1.enable_v2_behavior()

# Download the LSAT dataset and setup the required filepaths.
_DATA_ROOT = tempfile.mkdtemp(prefix='lsat-data')
_DATA_PATH = 'https://storage.googleapis.com/lawschool_dataset/bar_pass_prediction.csv'
_DATA_FILEPATH = os.path.join(_DATA_ROOT, 'bar_pass_prediction.csv')

data = urllib.request.urlopen(_DATA_PATH)

_LSAT_DF = pd.read_csv(data)

# To simpliy the case study, we will only use the columns that will be used for
# our model.
_COLUMN_NAMES = [
  'dnn_bar_pass_prediction',
  'gender',
  'lsat',
  'pass_bar',
  'race1',
  'ugpa',
]

_LSAT_DF.dropna()
_LSAT_DF['gender'] = _LSAT_DF['gender'].astype(str)
_LSAT_DF['race1'] = _LSAT_DF['race1'].astype(str)
_LSAT_DF = _LSAT_DF[_COLUMN_NAMES]

_LSAT_DF.head()

_LSAT_DF["gender"]=np.array([1 if x =='female' else 0 for x in _LSAT_DF["gender"]])
for race in np.unique(_LSAT_DF["race1"]):
    _LSAT_DF[race]=(_LSAT_DF["race1"]==race).astype(int)


## fit model
from sklearn.linear_model import LinearRegression

XGPA=np.array(_LSAT_DF[["gender",'asian', 'black', 'hisp', 'nan', 'other', 'white']])
GPA=np.array(_LSAT_DF[["ugpa"]])
reg=LinearRegression().fit(XGPA,GPA)

GPA_pred=reg.predict(XGPA)
GPA_error=GPA-GPA_pred


XLSAT=np.array(_LSAT_DF[["gender",'asian', 'black', 'hisp', 'nan', 'other', 'white']])
LSAT=np.array(_LSAT_DF[["lsat"]])
reg2=LinearRegression().fit(XLSAT,LSAT)

LSAT_pred=reg2.predict(XLSAT)
LSAT_error=LSAT-LSAT_pred

XFYA=np.array(_LSAT_DF[["gender",'asian', 'black', 'hisp', 'nan', 'other', 'white','ugpa','lsat']])
FYA=np.array(_LSAT_DF[["pass_bar"]])
reg3=LinearRegression().fit(XFYA,FYA)

FYA_pred=reg3.predict(XFYA)
FYA_error=FYA-FYA_pred


##
def IndividualExpectation(ind=0,bins=50):
    yICE=[]
    yIIE=[]
    for x in [0,1]:
        data=np.expand_dims(XFYA[ind].copy(),axis=0)
        data[0,0]=x
        yICE.append(reg3.predict(data))
        X=np.expand_dims(XLSAT[ind].copy(),axis=0)
        X[0,0]=x
        data[0,-2]=reg.predict(X)+GPA_error[ind]
        data[0,-1]=reg2.predict(X)+LSAT_error[ind]
        yIIE.append(reg3.predict(data)+FYA_error[ind])
    yICE=np.asarray(yICE).flatten()
    yIIE=np.array(yIIE).flatten()
    plt.figure('IIE')
    mpl.rc("figure", figsize=(9, 6))
    plt.plot([0,1],yICE,marker="o", markersize=10,linewidth=2,label="ICE")
    plt.plot([0,1],yIIE,marker="o", markersize=10,linewidth=2,label="IIE")
    plt.plot(XFYA[ind][0], FYA[ind], marker="o", markersize=10, markeredgecolor="red")
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(loc="right",fontsize=25)
    plt.xlabel('X2',fontsize=25)
    plt.title("Comparation between ICE and IIE",fontsize=25)
    plt.savefig('./IIE.png',dpi=200)

def PDPCPDP(bins=50):
    yICE=[]
    yIIE=[]
    for x in [0,1]:
        data=XFYA.copy()
        data[:,0]=x
        yICE.append(reg3.predict(data))
        X=XLSAT
        X[:,0]=x
        data[:,-2]=(reg.predict(X)+GPA_error).flatten()
        data[:,-1]=(reg2.predict(X)+LSAT_error).flatten()
        yIIE.append(reg3.predict(data)+FYA_error)
        
    yICE=np.concatenate(yICE,axis=1)
    yIIE=np.concatenate(yIIE,axis=1)
    
    yPDP=np.mean(yICE,axis=0)
    yCPDP=np.mean(yIIE,axis=0)
    
    plt.figure('CPDP')
    mpl.rc("figure", figsize=(9, 6))
    plt.plot([0,1],yPDP,marker="o", markersize=10,linewidth=2,label="PDP")
    plt.plot([0,1],yCPDP,marker="o", markersize=10,linewidth=2,label="CPDP")
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(loc="upper right",fontsize=25)
    plt.xlabel('X2',fontsize=25)
    plt.title("Comparation between PDP and CPDP",fontsize=25)
    plt.savefig('./CPDP.png',dpi=200)




IndividualExpectation(ind=0,bins=50)
PDPCPDP(bins=50)
