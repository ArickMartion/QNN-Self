from mindquantum.core.gates import *
from mindquantum.core.circuit import *
from mindquantum.simulator import Simulator
from mindquantum.core import Measure
from mindspore import Tensor

import numpy as np
from numpy import pi,cos,sin
import time

import copy


import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

"""1. Define the QNN class"""
class QNN():
    def __init__(self,QCircuit, params_name_dict, shots=1024, measure_type="state_vector", 
                 interpret=None,optimizer=None):
        self.params_name=params_name_dict["params_name"] # All parameters
        self.encode_params_name=params_name_dict["encode_params_name"] # Parameters used for encoding
        self.weight_params_name=params_name_dict["weight_params_name"] # Parameters used for computation
        self.valid_params_name=self.encode_params_name.copy()+self.weight_params_name.copy()
        
        self.params={key:0 for key in self.params_name}
        self.init_params=self.params.copy()
        
        self.shots=shots
        self.results=dict()
        self.measure_type=measure_type
        self.interpret=interpret # Transform the default sampling results
        
        self.qc=QCircuit
        self.sim = Simulator('mqvector', self.qc.n_qubits) 
        
        self.grad=None
        self.grad_combined=None
        
        self.optimizer=optimizer
        
        
    def initialize_parameters(self,params=None,random_seed=1,random_type="normal"):
        np.random.seed(random_seed)
        """Function: Initialize the parameters of the circuit"""
        if params is None:
            for key in self.valid_params_name:
                if random_type=="normal":
                    self.init_params[key]=np.random.normal(loc=0,scale=0.5)  
                elif random_type=="uniform":
                    self.init_params[key]=np.random.rand()*np.pi   
        else:
            self.init_params=params.copy
        self.params=self.init_params.copy()
        return self.init_params
    
    def forward(self,input_data=None,params=None,shots=None):
        """ Forward propagation: data dimension (batch_size, 2**n_qubits)"""
        if input_data is None:
            input_data=np.zeros((1,len(self.encode_params_name))).tolist()
        if params is None:
            params=self.params.copy()
        if shots is None:
            shots=self.shots

        Probs=[]
        for n in range(len(input_data)):
            params_ls=params.copy()
            for idx,name in enumerate(self.encode_params_name):
                if name in self.weight_params_name:
                    params_ls[name] += input_data[n][idx]
                else:
                    params_ls[name]=input_data[n][idx]

            P_dict=params_ls.copy()

            """Run"""

            if self.measure_type=="sampling":
                self.sim.reset() # Reset the Simulator
                results=self.sim.sampling(self.qc, P_dict, shots=shots)
                self.results=copy.deepcopy(results)
                results=results.data

                #print("self.measure_type",sampling)
                """# Process the results"""
                counts=[]
                n_bits=len(list(results.keys())[0])
                for k in range(0,2**(n_bits)):
                    key=bin(k)[2:].zfill(n_bits)
                    if n_bits>4:
                        if key[0]=="1":
                            continue
                    if key not in results.keys():
                        counts.append(0)
                    else:
                        counts.append(results[key])
                        
                #print(n_bits,n)
                counts=np.array(counts)#[0:16]
                Probs.append(counts/(1e-6+sum(counts)))

            elif self.measure_type=="state_vector":
                #print("self.measure_type",self.measure_type)
                
                self.sim.reset() # Reset the Simulator
                self.sim.apply_circuit(self.qc, pr=P_dict)
                results=self.sim.get_qs(False)
                results=abs(results)**2
                results=results#[0:16]
                results=results/sum(results)
                
                if self.interpret!=None:
                    results=self.interpret(results)
                
                self.results=copy.deepcopy(results)
                Probs.append(results)
                
                if np.isnan(results[0]):
                    print(P_dict.values())
                    print(results)

        Probs=np.array(Probs)

        return Probs
    
    
    def backward(self,input_data=None,params=None,requires_grad=True,shots=None,y_tar = None,
                 cost_fn = None,grad_method = None,label_qubits = [6,7]):
        """Backward propagation: data dimension (batch_size, output_shape, num_weights)"""
        if input_data is None:
            input_data=np.zeros((1,len(self.encode_params_name))).tolist()
        if params is None:
            params=self.params.copy()
        if shots is None:
            shots=self.shots
        if grad_method is None:
            grad_method = self.optimizer.grad_method
            
        optim = self.optimizer.optim
        
        num_weights=len(self.weight_params_name)
        
        grads=[]
        
        if optim in ["Adam","RMSprop","AMSGrad"]:
            for weight_name in self.weight_params_name:
                plus_params=copy.deepcopy(params)
                minus_params=copy.deepcopy(params)

                if grad_method == "parameter_shift":
                    plus_params[weight_name]+=np.pi/2
                    minus_params[weight_name]-=np.pi/2
                    eps = 1
                    #print(plus_params[weight_name],minus_params[weight_name])
                else:
                    plus_params[weight_name]+=0.2
                    minus_params[weight_name]-=0.2
                    eps = 0.2

                
                
                plus_dist=self.forward(input_data,params=plus_params,shots=shots)
                minus_dist=self.forward(input_data,params=minus_params,shots=shots)

                if cost_fn is None:
                    grad=(plus_dist-minus_dist)/(2*eps)
                else:
                    grad = (cost_fn(plus_dist, y_tar,label_qubits) - cost_fn(minus_dist, y_tar,label_qubits))/(2*eps)
                grads.append(grad)
                
            if cost_fn is not None:
                return np.array(grads).reshape(-1,)
            else:
                grads = np.array(grads)
                grads = np.transpose(grads, (1, 2, 0)) #(num_weights,batch_size, output_shape)->(batch_size, output_shape, num_weights)
                return grads
                
        if optim in ["SPSA","SPSA-Adam","SPSA-RMSprop","SPSA-AMSGrad"]:
            n_weights = len(self.weight_params_name)
            n_samples = self.optimizer.spsa_samples # Sample multiple times to improve the accuracy of gradient estimation
            
            ak = self.optimizer.ak
            ck = self.optimizer.ck
                
            grads_ave = []
            
            for k in range(n_samples):
                # Generate a random perturbation vector (values ±1)
                delta = 2 * np.random.randint(0, 2, n_weights) - 1

                # Compute the loss function on both sides
                plus_params=copy.deepcopy(params)
                minus_params=copy.deepcopy(params)

                for i,weight_name in enumerate(self.weight_params_name):
                    #print(plus_params[weight_name],delta[i])
                    plus_params[weight_name] += ck * delta[i]
                    minus_params[weight_name] -= ck * delta[i]

                plus_dist = torch.tensor(self.forward(input_data,params=plus_params,shots=shots))
                minus_dist = torch.tensor(self.forward(input_data,params=minus_params,shots=shots))
                
                if cost_fn is not None:
                    grads = (cost_fn(plus_dist, y_tar, label_qubits) - cost_fn(minus_dist, y_tar, label_qubits))/(2*ck*delta)
                
                else:
                    grads = []
                    for i in range(len(plus_dist)):
                        grads.append([])
                        for j in range(len(plus_dist[0])):
                            grad = (plus_dist[i][j]-minus_dist[i][j])/(2*ck*delta)
                            grads[-1].append(grad)
                    
                grads = np.array(grads)
                
                if self.optimizer.optim == "SPSA":
                    grads *= ak

                grads_ave.append(grads)
                
            grad_ave = np.mean(np.array(grads_ave),axis = 0)
            
            if cost_fn is None:
                grad_ave = np.transpose(grad_ave, (1, 2, 0))
 
            return grad_ave
    
    def combination_grad(self,grads_ls):
        """Function: Use backpropagation to compute the final gradient
           grads_ls[0]：[batch_size, xn]
           grads_ls[1]: [batch_size, xn, wn]->g0xg1=[batch_size, wn]
           grads_ls[2]: [batch_size, wn, xn-1]
           grads_ls[3]: [batch_size, xn-1, wn]
        """
        grad_combined=grads_ls[0]
        for k in range(1,len(grads_ls)):
            grad_combined=np.matmul(grad_combined,grads_ls[k])

        grad_combined=np.mean(grad_combined, axis=0).reshape(-1,)

        self.grad_combined=grad_combined
        return grad_combined
            
    def step(self,grad_combined=None,optim=None):
        """Update parameters once"""
        if grad_combined is None:
            grad_combined=self.grad_combined
        if optim is None:
            optim=self.optimizer.optim
        
        grad_combined=np.real(grad_combined)
        
        if optim=="Adam":
            grad_=self.optimizer.Adam(grad_combined).copy()
        elif optim=="RMSprop":
            grad_=self.optimizer.RMSprop(grad_combined).copy()
        elif optim=="AMSGrad":
            grad_=self.optimizer.AMSGrad(grad_combined).copy()
        elif optim=="SPSA":
            grad_=self.optimizer.SPSA(grad_combined).copy()
  
        elif optim == "SPSA-Adam":
            grad_=self.optimizer.SPSA(grad_combined).copy()
            grad_=self.optimizer.Adam(grad_).copy()
        elif optim == "SPSA-RMSprop":
            grad_=self.optimizer.SPSA(grad_combined).copy()
            grad_=self.optimizer.RMSprop(grad_).copy()
        elif optim == "SPSA-AMSGrad":
            grad_=self.optimizer.SPSA(grad_combined).copy()
            grad_=self.optimizer.AMSGrad(grad_).copy()
            
        for idx,weight_name in enumerate(self.weight_params_name):
            self.params[weight_name]-=grad_[idx]
        return self.params
    
    
"""2. Define the optimizer class"""
class Optimizer():
    def __init__(self, learning_rate=None,beta=None,epsilon=None, optim="Adam", spsa_params = {"c":0.2, "min_ck": 0.1, "grad_decay":0.8, "if_grad_smooth":False,"A":30, "spsa_samples":1},beta_spsa=None, grad_method = "parameter_shift"):
        
        self.m_t=None
        self.v_t=None
        self.learning_rate=learning_rate
        self.beta=beta
        self.epsilon=epsilon
        self.optim=optim
        self.grad_method = grad_method
        
        self.c = spsa_params["c"]
        self.ak = self.learning_rate
        self.A = spsa_params["A"] # SPSA stabilization coefficient
        self.ck = self.c
        self.min_ck = spsa_params["min_ck"]
        self.grad_decay = spsa_params["grad_decay"]
        self.if_grad_smooth = spsa_params["if_grad_smooth"]
        self.spsa_samples = spsa_params["spsa_samples"]
        self.beta_spsa = beta_spsa
        
        self.t = 0 # Optimization step
        self.grad_buffer =[]
        self.grad_smooth =None
        
    def RMSprop(self,grad,learning_rate=0.1,beta=0.99,epsilon=1e-10):
        """
        Implement the RMSProp optimizer

        Parameters:
            grad: gradient function, receives the current gradient.
            learning_rate: learning rate.
            beta: momentum parameter.
            epsilon: numerical stability constant, usually very small.

        Returns:
            grad_: optimized gradient.
        """
        if self.learning_rate!=None:
            learning_rate=self.learning_rate
        if self.beta is not None:
            beta=self.beta
        if self.epsilon is not None:
            epsilon=self.epsilon
        
        if self.v_t is None:
            v_t=np.zeros_like(grad)
        else:
            v_t = self.v_t.copy()
                
        v_t=beta*v_t+(1-beta)*(grad**2)
        v_t_hat = v_t / (1 - beta)
        grad_=learning_rate*grad/(np.sqrt(v_t_hat)+epsilon)
        
        self.v_t=v_t.copy()
        return grad_
    
    
    def Adam(self,grad=None,learning_rate=0.1, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """
        Implement the Adam optimizer

        Parameters:
            grad: gradient function, receives the current gradient.
            learning_rate: learning rate.
            beta1: momentum parameter.
            beta2: second-moment parameter.
            epsilon: numerical stability constant, usually very small.

        Returns:
            grad_: optimized gradient.
        """
        if self.learning_rate!=None:
            learning_rate=self.learning_rate
        if self.beta is not None:
            beta1=self.beta[0]
            beta2=self.beta[1]
        if self.epsilon is not None:
            epsilon=self.epsilon
            
        if self.m_t is None:
            m_t=np.zeros_like(grad)
            v_t=np.zeros_like(grad)
        else:
            m_t = self.m_t.copy()
            v_t = self.v_t.copy()
        

        m_t=beta1*m_t+(1 - beta1) * grad
        v_t=beta2*v_t+(1-beta2)*(grad**2)

        m_t_hat = m_t / (1 - beta1)
        v_t_hat = v_t / (1 - beta2)

        grad_=learning_rate*m_t_hat/(np.sqrt(v_t_hat)+epsilon)

        self.m_t=m_t.copy()
        self.v_t=v_t.copy()

        return grad_
    
    
    def AMSGrad(self,grad=None,learning_rate=0.1, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """
        Implement the Adam optimizer

        Parameters:
            grad: gradient function, takes the current gradient.
            learning_rate: learning rate.
            beta1: momentum parameter.
            beta2: second-moment parameter.
            epsilon: numerical stability constant, usually very small.

        Returns:
            grad_: optimized gradient.
        """
        if self.learning_rate!=None:
            learning_rate=self.learning_rate
        if self.beta is not None:
            beta1=self.beta[0]
            beta2=self.beta[1]
        if self.epsilon is not None:
            epsilon=self.epsilon
            
        if self.m_t is None:
            m_t=np.zeros_like(grad)
            v_t=np.zeros_like(grad)
            v_hat_max = np.zeros_like(grad)
        else:
            m_t = self.m_t.copy()
            v_t = self.v_t.copy()
            v_hat_max = self.v_hat_max
        

        m_t=beta1*m_t+(1 - beta1) * grad
        v_t=beta2*v_t+(1-beta2)*(grad**2)

        m_t_hat = m_t / (1 - beta1)
        v_t_hat = v_t / (1 - beta2)
        
        v_hat_max = np.maximum(v_hat_max, v_t_hat)

        grad_=learning_rate*m_t_hat/(np.sqrt(v_hat_max)+epsilon)

        self.m_t=m_t.copy()
        self.v_t=v_t.copy()
        self.v_hat_max = v_hat_max.copy()

        return grad_
    
    
    def SPSA(self, grad=None,learning_rate=0.1, c=0.1, alpha=0.602, gamma=0.101):
        if self.learning_rate!=None:
            learning_rate=self.learning_rate
        if self.beta_spsa is not None:
            alpha=self.beta_spsa[0] # learning rate decay coefficient
            gamma=self.beta_spsa[1] # Perturbation decay coefficient
            
        if self.c is not None:
            c = self.c
        
            
        ak = learning_rate/(self.t + 1+self.A)**alpha
        ck = c/(self.t + 1)**gamma
        
        self.ak = ak
        self.ck = max(ck,self.min_ck)
        self.t += 1
        
        if self.if_grad_smooth == True:
            if self.grad_smooth is None:
                self.grad_smooth = copy.deepcopy(grad)
            else:
                self.grad_smooth = self.grad_decay*self.grad_smooth + (1-self.grad_decay)*grad

            grad_ = self.grad_smooth 
            
        else:
            grad_ = copy.deepcopy(grad)

        return grad_
   