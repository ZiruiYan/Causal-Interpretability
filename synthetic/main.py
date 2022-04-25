#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 13:01:00 2022

@author: rain
"""
import numpy as np
import matplotlib.pyplot as plt
import os
# os.chdir("/Users/rain/Desktop/Learning/TML")
# from model import self_net
from model_seperate import net_x2, net_x3,net_y
import torch
import torch.nn as nn

np.random.seed(2022)
torch.manual_seed(2022)


epsilon1=np.random.uniform(low=-10, high=10,size=(50000,1))

# epsilon1=np.concatenate((np.random.normal(loc=1,size=(250000,1)),np.random.normal(loc=-1,size=(250000,1))))
# epsilon1=np.random.normal(loc=1,size=(5000000,1))
epsilon2=np.random.normal(scale=0.01,size=(50000,1))
# epsilon2=np.zeros((5000000,1))
epsilon3=np.random.normal(scale=0.01,size=(50000,1))
epsilon4=np.random.normal(scale=0.01,size=(50000,1))

def sigmoid(x):
    return 1/(1+np.e**(-x))

x1=epsilon1
x2=10*sigmoid(x1)-5+epsilon2
x3=10*sigmoid(-x1-x2)-5+epsilon3
y=10*sigmoid(x1+x2-x3)-5+epsilon4

x1=torch.from_numpy(x1)
x2=torch.from_numpy(x2)
x3=torch.from_numpy(x3)
y=torch.from_numpy(y)

model_x2=net_x2()
model_x2.double()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model_x2.parameters(),lr=0.1,weight_decay=0.0001)

for i in range(1000):
    out2 = model_x2(x1)
    loss = criterion(out2, x2)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (i+1)%10 ==0:
        print('Epoch:{}, Loss{:.5f}'.format(i+1,loss.item()))
        

for name, param in model_x2.named_parameters():
    if param.requires_grad:
        print(name, param.data)

x2_pred=model_x2(x1)
epsilon2_pred=(x2-x2_pred).detach().numpy()


model_x3=net_x3()
model_x3.double()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model_x3.parameters(),lr=0.1,weight_decay=0.0001)

for i in range(1000):
    out3 = model_x3(x1,x2)
    loss = criterion(out3, x3)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (i+1)%10 ==0:
        print('Epoch:{}, Loss{:.5f}'.format(i+1,loss.item()))
        

for name, param in model_x3.named_parameters():
    if param.requires_grad:
        print(name, param.data)

x3_pred=model_x3(x1,x2)
epsilon3_pred=x3-x3_pred

model_y=net_y()
model_y.double()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model_y.parameters(),lr=1,weight_decay=0.0001)

for i in range(2000):
    outy = model_y(x1,x2,x3)
    loss = criterion(outy, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (i+1)%10 ==0:
        print('Epoch:{}, Loss{:.5f}'.format(i+1,loss.item()))
        

for name, param in model_y.named_parameters():
    if param.requires_grad:
        print(name, param.data)

y_pred=model_y(x1,x2,x3)
epsilon4_pred=y-y_pred



from functions import IndividualExpectation
from functions import PDPCPDP

IndividualExpectation(x1,x2,x3,y,model_y,model_x3,epsilon3_pred,epsilon4_pred,epsilon3,epsilon4,ind=200,bins=50)
PDPCPDP(x1,x2,x3,y,model_y,model_x3,epsilon3_pred,epsilon4_pred,epsilon3,epsilon4,bins=50)



data=torch.cat((x1,x2,x3),axis=1)
feature=[1]

from alepytorch import ale_pytorch
import matplotlib as mpl
mpl.rc("figure", figsize=(9, 6))
ale, quantiles=ale_pytorch(model_y,
    data,
    feature,
    bins=100)

from alcepytorch import alce_pytorch
import matplotlib as mpl
mpl.rc("figure", figsize=(9, 6))
ale_alce, quantiles_alce=alce_pytorch(model_y,
    model_x3,
    data,
    feature,
    epsilon3_pred,
    epsilon4_pred,
    bins=100)

from alcetrue import alce_true
import matplotlib as mpl
mpl.rc("figure", figsize=(9, 6))
ale_true, quantiles_true =alce_true(model_y,
    model_x3,
    data,
    feature,
    epsilon3,
    epsilon4,
    bins=100)


quantiles_center = (quantiles[1:] + quantiles[:-1]) / 2
plt.figure('ACLE')
mpl.rc("figure", figsize=(9, 6))
plt.plot(quantiles_center ,ale,linewidth=2,label="ALE")
plt.plot(quantiles_center ,ale_alce,linewidth=2,label="ALCE")
plt.plot(quantiles_center ,ale_true, linewidth=2,label="True")
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend(loc="upper left",fontsize=25)
plt.xlabel('X2',fontsize=25)
plt.title("Comparation between ALE and ALCE",fontsize=25)
plt.savefig('./ACLE.png',dpi=200)