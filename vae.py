# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 16:44:13 2020

@author: simoussi
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#import numpy as np
import pandas as pd
import json


import tensorflow as tf
import tensorflow_probability as tfp

tfk = tf.keras
tfkl = tf.keras.layers
tfpl = tfp.layers
tfd = tfp.distributions
tfkv= tfk.utils



'''
Load data
'''
X=pd.read_csv('data/bouche.csv',sep=',',decimal='.',index_col=0)
Y=pd.read_csv('data/occur.csv',sep=',',decimal='.',index_col=0)

with open('data/idx_train.json','r') as f:
    idx_train=json.load(f)

with open('data/idx_test.json','r') as f:
    idx_test=json.load(f)


train_dataset=Y.loc[idx_train]
eval_dataset=Y.loc[idx_test]

'''
Configure architecture size
'''

input_shape = Y.shape[1]
encoded_size = 16


'''
Prior network
'''
prior = tfd.Independent(tfd.Normal(loc=tf.zeros(encoded_size), scale=1),
                        reinterpreted_batch_ndims=1)


'''
Encoder network
'''

def encoder(in_config,encoded_size,paramsize):
    
    '''
    Collect inputs, embed categorical inputs and concatenate all to get features
    '''
    l_in=[]
    l_feat=[]
    
    for k in in_config.keys():
        if k=='num':
            in_num=tfk.Input(shape=(in_config.get(k).get('dim'),),dtype=tf.float32,name=in_config.get(k).get('id'))
            l_in.append(in_num)
            l_feat.append(in_num)
        
        if k=='cat':
            for cv in in_config.get(k):
                in_cat=tfk.Input(shape=(1,),dtype=tf.int32)
                emb_cat=tfkl.Flatten()(tfkl.Embedding(cv.get('mod'),cv.get('emb'))(in_cat))
                
                l_in.append(in_cat)
                l_feat.append(emb_cat)
                
    
    if len(l_feat)>1:       
        in_feat=tfkl.Concatenate()(l_feat)
        
    elif len(l_feat)==1:
        in_feat=l_feat[0]
        
    else:
        raise BaseException('ModelSpecificationError: check input data configuration')
    
    '''
    Map features to latent space, using a fully connected neural network
    '''
    
    latent_params=tfkl.Dense(units=paramsize,activation=None)(in_feat)
    
    return tfk.Model(l_in,latent_params)

def code_generator(encoded_size,prior,latent_params):    
    out_code=tfpl.MultivariateNormalTriL(
        encoded_size,
        activity_regularizer=tfpl.KLDivergenceRegularizer(prior))(latent_params)
    return(out_code)

'''
Decoder network
'''
def decoder(in_config,code):
    '''
    Map latent variables to output features:
        - Shared layers: among all outputs
        - Specific layers: used to map individual tasks to their outputs
    '''
    logit=tfkl.Dense(units=input_shape)(code)

    
    return logit
  

'''
Full model
'''

in_config={'num':{'id':'num','dim':3,'act':'relu'},'cat':[{'id':'cat1','mod':5,'emb':2},{'id':'cat2','mod':7,'emb':4}]}
paramsize=tfpl.MultivariateNormalTriL.params_size(encoded_size)
encoder_net=encoder(in_config,encoded_size,paramsize)

recogn_net=tfk.Model(encoder_net.inputs,code_generator(encoded_size,prior,encoder_net.output))
 
vae = tfk.Model(inputs=recogn_net.inputs,
                outputs=decoder(input_shape,recogn_net.output))

vae.summary() 

'''
Visualize architecture
'''
tfkv.plot_model(vae)
vae.summary()


negloglik = lambda x, rv_x: -rv_x.log_prob(x)

vae.compile(optimizer=tf.optimizers.Adam(learning_rate=1e-3),
            loss=negloglik)

'''
Train
'''
_ = vae.fit(train_dataset,
            epochs=15,
            validation_data=eval_dataset)