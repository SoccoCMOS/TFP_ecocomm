# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 15:59:42 2020

@author: saras
"""


from lego_blocks import *
from eval_functions import *
import tensorflow_addons as tfa
import numpy as np

 
seed=1234
tf.random.set_seed(seed)


class TaxaEmbedding(tfkl.Layer):
    def __init__(self, output_dim,name, **kwargs):
       super(TaxaEmbedding, self).__init__(**kwargs)
       self.output_dim = output_dim
       self.vname=name

    def build(self, input_shapes):
       self.kernel = self.add_weight(name=self.vname, 
                                     shape=self.output_dim, 
                                     initializer='uniform', 
                                     trainable=True)       
       
       super(TaxaEmbedding, self).build(input_shapes)  

    def call(self, inputs):
       return self.kernel

    def compute_output_shape(self):
       return self.output_dim

class AssociationEmbedding(tfkl.Layer):
    def __init__(self, output_dim,sym,reg,mask, **kwargs):       
       ## Get dimension
       self.output_dim = output_dim
       self.is_symmetric=sym
       ## Prepare mask for diagonals
       self.mask=tf.ones((self.output_dim,self.output_dim))
       if mask:
           self.mask-=tf.eye(self.output_dim)
       self.assoc_reg=tfkr.l1_l2(l1=reg[0],l2=reg[1])
       #self.var_assoc=var_assoc

       super(AssociationEmbedding, self).__init__(**kwargs)

    def build(self, input_shape):
       ## Parameters are embeddings of response and effect
       emb_dim=tf.constant([self.output_dim,input_shape[2]])
       if self.is_symmetric:
           self.eff_emb=self.resp_emb=TaxaEmbedding(emb_dim,'latent')(None)
       else:
           self.eff_emb=TaxaEmbedding(emb_dim,'effect')(None)
           self.resp_emb=TaxaEmbedding(emb_dim,'response')(None)
       
       self.assoc=tfk.backend.dot(self.resp_emb,tf.transpose(self.eff_emb))
       
       ### Add regularization constraints over associations
       ## Sparsity inducing constraint
       self.add_loss(lambda: self.assoc_reg(self.assoc))
       
       ## Other constraints
       
       super(AssociationEmbedding, self).build(input_shape)  

    def call(self, loadings):  
        ### inputs has dimension [n,1,d] for shared loadings or [n,m,d] for specific loadings per species
        ## alpha=[m,d], input=[n,1,d]   => [n,m,d]
        loaded_effect=tf.multiply(loadings,self.eff_emb)
        
        ## associations : [n,m,d] x [m,d] => [n,m,m]
        local_assoc=tfk.backend.dot(loaded_effect,tf.transpose(self.resp_emb))

        return tf.multiply(local_assoc,self.mask)

    def compute_output_shape(self,input_shape):
       return tf.constant([input_shape[0],self.output_dim,self.output_dim])


def gammaln(x):
    # fast approximate gammaln from paul mineiro
    # http://www.machinedlearnings.com/2011/06/faster-lda.html
    logterm = tfm.log (x * (1.0 + x) * (2.0 + x))
    xp3 = 3.0 + x
    return -2.081061466 - x + 0.0833333 / xp3 - logterm + (2.5 + x) * tfm.log (xp3)


def negbin_loss(y_true,y_pred):   ###uses y_pred as log probability or logit or applies softplus=relu to it beforehand
    r=y_pred[0]
    p=y_pred[1]
    logprob = gammaln(y_true + r) - gammaln(y_true + 1.0) -  \
                 gammaln(r) + r * tfm.log(r) + \
                 y_true * tfm.log(p+1E-6) - (r + y_true) * tfm.log(r + p)

    return tfk.backend.mean(logprob, axis=-1)


#### Activations ####
act_fn={'normal':tfk.activations.linear,
        'poisson2':tfk.activations.exponential,
        'poisson':tfk.activations.relu,
        'binomial':tfk.activations.sigmoid,
        'binomial2':probit,
        'categorical':tfk.activations.softmax,
        'negbin':tfk.activations.relu,
        'negbin2':tfk.activations.exponential
        }

#### Loss functions ####
loss_fn={'normal':tfk.losses.mean_squared_error,
        'poisson2':tfk.losses.poisson,
        'poisson':tfk.losses.poisson,
        'binomial':tfk.losses.binary_crossentropy,
        'binomial2':tfk.losses.binary_crossentropy,
        'categorical':tfk.losses.categorical_crossentropy,
        'negbin':negbin_loss,
        'negbin2':negbin_loss
        }

metric_fn={
    'regression':[tfa.metrics.RSquare()],
    'classification':[tfkm.BinaryAccuracy(),tfkm.AUC(),
                      tfkm.Precision(),tfkm.Recall(),#tss
                      tfkm.PrecisionAtRecall(recall=0.5),
                      tfkm.SensitivityAtSpecificity(specificity=0.5),
                      tfkm.TruePositives(),tfkm.FalsePositives(),
                      tfkm.TrueNegatives(),tfkm.FalseNegatives()],
    'mclassification':[tfkm.CategoricalAccuracy()],
    'count':[tfkm.MeanSquaredError(),#tfkm.MeanAbsolutePercentageError(),
             tfkm.MeanAbsoluteError(),poisson_dev]
    }

class EcoAssocNet(object):
    def __init__(self,model_name="model",shared_config=None,hsm_config=None,im_config=None):
        self.model_name=model_name
        self.hsm_config=hsm_config
        self.im_config=im_config
        self.shared_config=shared_config
        
        self.d=self.shared_config['latent_dim']
        self.m=self.shared_config['pool_size']
        self.var_assoc=(len(self.im_config['input'])>0)
        
        self.dist=shared_config['dist']
        
    def create_architecture(self,offsets=None):
        ## generate abiotic and biotic covariates
        self.gen_covs()
        
        ## Abiotic response
        x_abio=self.fe_env_net(self.env_feat)
        
        ## Here, replace following dense layer by another network that yields regression parameters given traits
        abio_resp=tfkl.Dense(self.m,use_bias=self.im_config['fit_bias']=='intercept')(x_abio)
        
        ## Generate biotic response
        if self.var_assoc:
            x_bio=tf.expand_dims(self.fe_bio_net(self.bio_feat),axis=1)
        else:
            x_bio=tfkl.Lambda(lambda x: tf.ones((tf.shape(x)[0],1,self.d)),name=self.model_name+'_loadings')(abio_resp)
        
        assoc=self.association(x_bio)
        bio_resp=tfkl.Dot(axes=1)([self.counts,assoc])
        
        ### Aggregate abiotic and biotic effects
        drivers=[abio_resp,bio_resp]
        if self.im_config['fit_bias']=='offset':
            drivers+=[tf.expand_dims(tf.constant(offsets,dtype=tf.float32),0)]
        
        logits=tfkl.Add(name=self.model_name+'_out')(drivers)
        pred=tfkl.Activation(act_fn.get(self.dist[0]))(logits)
        
        self.eta_model=tfk.Model(self.inputs,logits) 
        self.pred_model=tfk.Model(self.inputs,pred)
    
    def gen_covs(self):
        self.inputs=[]
        self.env_in,self.env_feat=custom_input(self.hsm_config['input'],concat_feat=True,
                         model_name=self.model_name+'_in_env')
                
        self.counts=tfkl.Input((self.m,),name=self.model_name+'_count')     
        
        self.fe_env_net=fc_nn(name=self.model_name+'_fe_env',
                          archi=self.hsm_config['archi'],
                          reg=self.hsm_config['reg'])
        
        self.inputs=[self.env_in,self.counts]
        
        if self.var_assoc:
            self.bio_in,self.bio_feat=custom_input(self.im_config['input'],concat_feat=True,
                             model_name=self.model_name+'_in_bio')
            self.fe_bio_net=fc_nn(name=self.model_name+'_fe_bio',
                              archi=self.im_config['archi'],
                              reg=im_config['reg'])
            
        self.association=AssociationEmbedding(self.m,self.im_config['sym'],self.shared_config['reg'],True)
        
    
    def compile_model(self,on_logit=False,opt='adagrad',mets=[]):
        self.on_logit=on_logit
        
        if self.dist[1] is not None:
            loss=loss_fn.get(self.dist[0])(self.dist[1])
            
        else:
            loss=loss_fn.get(self.dist[0])
            
        if on_logit:
            self.eta_model.compile(optimizer=opt,loss=loss,metrics=mets)
            
        else:
            self.pred_model.compile(optimizer=opt,loss=loss,metrics=mets)
            
    
    def fit_model(self,trainset,validset,train_config,cbk,vb):
        self.pred_model.fit(x=[trainset['x'],trainset['y']],y=trainset['y'],
                            validation_data=([validset['x'],validset['y']],validset['y']),
                            batch_size=train_config['bsize'],epochs=train_config['epoch'],
                            callbacks=cbk,verbose=vb)
        
        

'''
TODO: 
    1. Add support for gaussian distribution
    2. Add support for negative binomial distribution
    3. Add support for multinomial and dirichlet distribution
    4. Add convergence check with tol and max_iter
    5. Add model selection with loglikelihood, BIC, eBIC, stars, bootstrap
'''        
        
