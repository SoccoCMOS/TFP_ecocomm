# -*- coding: utf-8 -*-
"""
Created on Mon May 25 15:37:57 2020

@author: @gokceneraslan
"""


import tensorflow as tf 

# We need a class (or closure) here,
# because it's not possible to
# pass extra arguments to Keras loss functions
# See https://github.com/fchollet/keras/issues/2121

# dispersion (theta) parameter is a scalar by default.
# scale_factor scales the nbinom mean before the 
# calculation of the loss to balance the
# learning rates of theta and network weights

tfm=tf.math
class NB(object):
    def __init__(self, theta_var=None,scale_factor=1.0, scope='nbinom_loss/'):
        
        # for numerical stability
        self.eps = 1e-6
        self.scale_factor = scale_factor

        # keep a reference to the variable itself
        self.theta_variable = theta_var

        # to keep dispersion always non-negative
        #self.theta = tf.nn.softplus(theta_var)
           
    def loss(self, y_true, y_pred, reduce=True):
        scale_factor = self.scale_factor
        eps = self.eps

        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32) * scale_factor
        
        theta = 1.0/(tf.nn.softplus(self.theta_variable)+eps)

        t1 = -tfm.lgamma(y_true+theta+eps) 
        t2 = tfm.lgamma(theta+eps)
        t3 = tfm.lgamma(y_true+1.0) 
        t4 = -(theta * (tfm.log(theta+eps)))
        t5 = -(y_true * (tfm.log(y_pred+eps)))
        t6 = (theta+y_true) * tfm.log(theta+y_pred+eps)      
        final = t1 + t2 + t3 + t4 + t5 + t6
        
        if reduce:
            final = tf.reduce_mean(final)
            
        return final