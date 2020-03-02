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
encoder = tfk.Sequential([
    tfkl.InputLayer(input_shape=input_shape),
    tfkl.Dense(tfpl.MultivariateNormalTriL.params_size(encoded_size),
               activation=None),
    tfpl.MultivariateNormalTriL(
        encoded_size,
        activity_regularizer=tfpl.KLDivergenceRegularizer(prior)),
])
    

'''
Decoder network
'''

decoder = tfk.Sequential([
    tfkl.InputLayer(input_shape=[encoded_size]),
    tfkl.Dense(tfpl.MultivariateNormalTriL.params_size(encoded_size),
               activation=None),
    tfpl.IndependentBernoulli(input_shape, tfd.Bernoulli.logits),
])
  

'''
Full model
'''
    
vae = tfk.Model(inputs=encoder.inputs,
                outputs=decoder(encoder.outputs[0]))


negloglik = lambda x, rv_x: -rv_x.log_prob(x)

vae.compile(optimizer=tf.optimizers.Adam(learning_rate=1e-3),
            loss=negloglik)

'''
Train
'''
_ = vae.fit(train_dataset,
            epochs=15,
            validation_data=eval_dataset)