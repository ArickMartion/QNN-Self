from mindquantum.core.gates import *
from mindquantum.core.circuit import *
from mindquantum.simulator import Simulator
from mindquantum.core import Measure
from mindspore import Tensor

import numpy as np
from numpy import pi,cos,sin
import time

import copy
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

"""1. Compute the expectation values of qubits"""

def qubit_exps(dist, qs=[2, 3]):
    nq = int(np.log2(dist.shape[1]))  
    
    q_exps = torch.zeros((dist.shape[0], len(qs)), dtype=dist.dtype, device=dist.device)
    
    for idx, q in enumerate(qs):
        d_plus = 2**q
        
        # Create a mask to replace slicing operations
        mask_minus = torch.zeros(dist.shape[1], dtype=torch.bool, device=dist.device)
        mask_plus = torch.zeros(dist.shape[1], dtype=torch.bool, device=dist.device)
        
        # Construct a mask for subtraction
        start = 0
        while start < dist.shape[1]:
            mask_minus[start:start+d_plus] = True
            start += 2 * d_plus
            
        # Construct a mask for addition
        start = d_plus
        while start < dist.shape[1]:
            mask_plus[start:start+d_plus] = True
            start += 2 * d_plus
        
        # Perform operations using the mask to preserve gradients
        q_exps[:, idx] = torch.sum(dist[:, mask_plus], dim=1) - torch.sum(dist[:, mask_minus], dim=1)
    
    return q_exps
        
"""2. Obtain the predicted data labels"""
def label_predict( x, qs=[2,3], data_type = "exp" ):
    if data_type == "dist":
        q_exps=qubit_exps( x, qs )
    elif data_type == "exp":
        q_exps=x
    labels=torch.zeros(q_exps.size()[0])
    
    for l in range(len(labels)):
        labels[l]=torch.argmax(q_exps[l])
        
    return labels.detach().numpy()

"""3. Perform one-hot encoding"""
def one_hot_encoding(y_tar,y_gen):
    # Check if y_true is one-hot encoded
    if y_tar.ndim == 1 or y_tar.shape[1] == 1:
        # Convert class index format to one-hot encoding
        y_tar = torch.eye(y_gen.shape[1])[y_tar]
    return y_tar
    
"""4.softmax"""
def softmax(X):
    #print(torch.max(X, dim=1).values)
    #exp_X = torch.exp(X - torch.max(X, dim=1).values.view(-1,1))  # 
    exp_X = torch.exp(X)  # Stabilization processing
    return exp_X / torch.sum(exp_X, dim=1).view(-1,1)
    
"""5.loss function"""


def loss_fn(y_tar, y_gen, loss_type="CrossEntropy"):
    
    y_tar=one_hot_encoding(y_tar, y_gen)
    y_tar=torch.clip(y_tar, 0, 1)
    
    if loss_type=="CrossEntropy":
        y_gen=softmax(y_gen)
        
        loss=-torch.sum(y_tar*torch.log(y_gen),dim=1)
        loss=torch.mean(loss)
        
        #print(y_tar)
        #print(y_gen)
        
    elif loss_type=="l2norm":
        
        y_gen=y_gen/torch.sum(y_gen, dim=1).view(-1,1)
        loss=torch.mean(torch.sum((y_gen-y_tar)**2, dim=1))
    
    return loss

"""6. Compute accuracy"""
def Acc(y_tar, qnn_out, qs=[2,3], data_type = "dist"):
    label_tar=y_tar
    label_gen=label_predict( qnn_out, qs ,data_type)
    
    #print(label_gen)
    
    acc=0
    for k in range(len(label_tar)):
        if abs(label_tar[k]-label_gen[k])<=0.01:
            acc+=1
    acc/=len(label_tar)
    return acc

"""7. Compute the one-hot encoded y output"""
def QClassification_y_gen( qnn_out, qs=[2,3], retain_grad=False ):
    qnn_out=(qnn_out[:,0:16]+1e-12)/torch.sum(qnn_out[:,0:16]+1e-12,dim=1).view(-1,1)
    if retain_grad==True:
        qnn_out.retain_grad()
    y_gen=qubit_exps( qnn_out, qs)/torch.tensor(2)+torch.tensor(0.5)
    return y_gen