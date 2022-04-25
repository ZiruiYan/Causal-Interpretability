import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl

def sigmoid(x):
    return 1/(1+np.e**(-x))

def IndividualExpectation(x1,x2,x3,y,model_y,model_x3,epsilon3_pred,epsilon4_pred,epsilon3,epsilon4,ind=0,bins=50):
    maxmum=np.max(np.array(x2))
    minmum=np.min(np.array(x2))
    xrange=np.linspace(minmum,maxmum,num=bins)
    yICE=[]
    yIIE=[]
    yTrue=[]
    for x in xrange:
        x2_int=torch.tensor(x).expand(1,1)
        yICE.append(model_y(x1[ind].expand(1,1),x2_int,x3[ind].expand(1,1)))
        yIIE.append(model_y(x1[ind].expand(1,1),x2_int,model_x3(x1[ind].expand(1,1),x2_int)+epsilon3_pred[ind])+epsilon4_pred[ind])
        x3_int = 10*sigmoid(-x1[ind]-x2_int)-5+epsilon3[ind]
        yTrue.append(10*sigmoid(x1[ind]+x2_int-x3_int)-5+epsilon4[ind])
    yICE=np.asarray(yICE)
    yIIE=np.array(yIIE)
    yTrue=np.array(yTrue)
    plt.figure('IIE')
    mpl.rc("figure", figsize=(9, 6))
    plt.plot(xrange,yICE,linewidth=2,label="ICE")
    plt.plot(xrange,yIIE,linewidth=2,label="IIE")
    plt.plot(xrange,yTrue,linewidth=2,label="True")
    plt.plot(x2[ind], y[ind], marker="o", markersize=10, markeredgecolor="red")
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(loc="lower right",fontsize=25)
    plt.xlabel('X2',fontsize=25)
    plt.title("Comparation between ICE and IIE",fontsize=25)
    plt.savefig('./IIE.png',dpi=200)


def PDPCPDP(x1,x2,x3,y,model_y,model_x3,epsilon3_pred,epsilon4_pred,epsilon3,epsilon4,bins=50):
    maxmum=np.max(np.array(x2))
    minmum=np.min(np.array(x2))
    xrange=np.linspace(minmum,maxmum,num=bins)
    yICE=[]
    yIIE=[]
    yTrue=[]
    num=x2.shape[0]
    for x in xrange:
        x2_int=torch.tensor(np.repeat(x,num).reshape(num,1))
        yICE.append(model_y(x1,x2_int,x3))
        yIIE.append(model_y(x1,x2_int,model_x3(x1,x2_int)+epsilon3_pred)+epsilon4_pred)
        x3_int = 10*sigmoid(-x1-x2_int)-5+epsilon3
        yTrue.append(10*sigmoid(x1+x2_int-x3_int)-5+epsilon4)
        # print(yTrue)
    yICE=torch.cat(yICE,dim=1).detach().numpy()
    yIIE=torch.cat(yIIE,dim=1).detach().numpy()
    yTrue=torch.cat(yTrue,dim=1).numpy()
    plt.figure('CPDP')
    mpl.rc("figure", figsize=(9, 6))
    yPDP=np.mean(yICE,axis=0)
    yCPDP=np.mean(yIIE,axis=0)
    yTrue=np.mean(yTrue,axis=0)
    plt.plot(xrange,yPDP,linewidth=2,label="ICE")
    plt.plot(xrange,yCPDP,linewidth=2,label="IIE")
    plt.plot(xrange,yTrue,linewidth=2,label="True")
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(loc="lower right",fontsize=25)
    plt.xlabel('X2',fontsize=25)
    plt.title("Comparation between PDP and CPDP",fontsize=25)
    plt.savefig('./CPDP.png',dpi=200)
