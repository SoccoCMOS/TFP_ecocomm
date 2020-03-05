# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 10:31:31 2020

@author: saras

Adapted from Di Chen et al 2018 (E2EL for the DeepMVP)
"""

from lego_blocks import *
import numpy as np

class DeepMVP(object):
    def __init__(self,model_name='dmvp',in_config=None,fe_config=None,out_config=None,m=1,r=3):
        self.model_name=model_name
        self.dmvp=None
        
        self.in_config=in_config
        self.fe_config=fe_config
        self.out_config=out_config
        self.ntasks=m
        self.rank=r
        
        self.eps2=tf.constant(1e-6*2.0**(-100), dtype="float64")
        self.eps1=tf.constant(1e-6, dtype="float32")
        self.eps3=1e-30
        
        
    def create_architecture(self):
        ###Input features ###
        l_in_E,l_E=custom_input(self.in_config,concat_feat=True,model_name='in_fe_'+self.model_name)
        
        ###Feature extraction###
        feat_net=mtl_output(self.fe_config,self.out_config,model_name='mu')
        
        ### Mean mu prediction => simple multispecies model ###
        r_mu=mu_net(l_E)
        
        '''
        Low-rank residuals
        '''        

        ### Covariance sigma estimation => tunable parameter ###
        r_sqrt_sigma=tf.Variable(
            initial_value=np.random.uniform(-np.sqrt(6.0/(self.ntasks+self.rank)), np.sqrt(6.0/(self.ntasks+self.rank)), (self.ntasks, self.rank)), 
            dtype=tf.float32, name='r_sqrt_sigma',trainable=True)

        '''
        Residual covariance  r_sigma and full covariance matrix sigma
        '''
		#compute the residual covariance matrix, which is guaranteed to be semi-positive definite
        r_sigma=tf.matmul(r_sqrt_sigma, tf.transpose(r_sqrt_sigma))
        sigma=r_sigma + tf.eye(self.ntasks)
		
        noise=tfpl.IndependentNormal(event_shape=(1,self.rank))(tf.constant([0.]*self.rank + [1.] * self.rank))
  
        sample_r=tf.matmul(noise,tf.transpose(r_sqrt_sigma))

        ### Output is probit of W
        Y_hat = tfd.Normal(0., 1.).cdf(sample_r)*(1-self.eps1)+self.eps1*0.5
        self.dmvp=tfk.Model(l_in_E,Y_hat)

    
    def plot_model(self):
        tfkv.plot_model(self.dmvp)
        
        


####################
        '''"Unit test"'''
####################        
in_feat=[('num',{'id':'env','dim':29})]
model_name='lombrics'
m=86
fe_config={'name':'fe','archi':{'nbnum':29,'nl':0,'nn':[],'activation':None},'reg':None}
taxa={'name':'taxa','archi':{'nbnum':29,'nl':1,'nn':[m],'activation':'relu'},'reg':None}
out_config=[ {'name':'mean','type':'binary','specific':taxa,'activation':'linear'}]

dmvp=DeepMVP(model_name,in_feat,fe_config,out_config)
self=dmvp 

self=dmvp   