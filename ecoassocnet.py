# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 15:59:42 2020

@author: saras
"""


from lego_blocks import *
from eval_functions import *
from NB import NB
import numpy as np

 
seed=1234
tf.random.set_seed(seed)


def compute_density(mat,th=0.01):
    return np.sum(np.abs(mat)>=th)/(mat.shape[0]*mat.shape[1])

def compute_effective_dimension(resp,eff,eps=1E-4):
    rdimuse=(np.max(np.abs(resp),axis=0)>eps)
    edimuse=(np.max(np.abs(eff),axis=0)>eps)
    return np.sum(rdimuse | edimuse)
                

class TaxaEmbedding(tfkl.Layer):
    def __init__(self, output_dim,name,norm, **kwargs):
       super(TaxaEmbedding, self).__init__(**kwargs)
       self.output_dim = output_dim
       self.vname=name
       self.norm=norm

    def build(self, input_shapes):
       self.kernel = self.add_weight(name=self.vname, 
                                     shape=self.output_dim, 
                                     initializer='uniform', 
                                     #regularizer=l1,
                                     constraint=tfk.constraints.UnitNorm(axis=1) if self.norm else None,
                                     trainable=True)       
       
       super(TaxaEmbedding, self).build(input_shapes)  

    def call(self, inputs):
       return self.kernel

    def compute_output_shape(self):
       return self.output_dim

class GlobalParam(tfkl.Layer):

    def __init__(self, output_dim,vname, **kwargs):
       self.output_dim = output_dim
       self.vname=vname
       super(GlobalParam, self).__init__(**kwargs)

    def build(self, input_shapes):
       self.kernel = tf.Variable(tf.random_uniform_initializer(minval=-1,maxval=1)(shape=[1,self.output_dim],dtype=tf.float32),
                                 name=self.vname, 
                                 trainable=True)
       
       super(GlobalParam, self).build(input_shapes)  

    def call(self, inputs):
       return inputs
    
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
       
       self.reg_emb=reg[2] if len(reg)>2 else True
       self.norm=reg[3] if len(reg)>3 else True 

       super(AssociationEmbedding, self).__init__(**kwargs)

    def build(self, input_shape):
       ## Parameters are embeddings of response and effect
       emb_dim=tf.constant([self.output_dim,input_shape[2]])
       if self.is_symmetric:
           self.eff_emb=self.resp_emb=TaxaEmbedding(emb_dim,'latent',self.norm)(None)
       else:
           self.eff_emb=TaxaEmbedding(emb_dim,'effect',self.norm)(None)
           self.resp_emb=TaxaEmbedding(emb_dim,'response',self.norm)(None)
       
       ### Add regularization constraints over associations
       ## Sparsity inducing constraint
       if self.reg_emb:
           self.add_loss(lambda: self.assoc_reg(self.resp_emb)+self.assoc_reg(self.eff_emb))
       else:
           self.add_loss(lambda: self.assoc_reg(tf.tensordot(self.resp_emb,self.eff_emb,axes=[1,1])))
       
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
        
    def create_architecture(self,offsets=None,mode='add'):
        ## generate abiotic and biotic covariates
        self.gen_covs()
        
        ## Abiotic response
        x_abio=self.fe_env_net(self.env_feat)
        
        ## Here, replace following dense layer by another network that yields regression parameters given traits
        abio_resp=tfkl.Dense(self.m,use_bias=self.im_config['archi']['fit_bias']=='intercept')(x_abio)
        
        ## Generate biotic response
        if self.var_assoc:
            x_bio=tf.expand_dims(self.fe_bio_net(self.bio_feat),axis=1)
        else:
            x_bio=tfkl.Lambda(lambda x: tf.ones((tf.shape(x)[0],1,self.d)),name=self.model_name+'_loadings')(abio_resp)
        
        assoc=self.association(x_bio)
        
        bio_resp=tfkl.Dot(axes=1)([self.counts,assoc])
        
        ### Aggregate abiotic and biotic effects
        drivers=[abio_resp,bio_resp]
        if self.im_config['archi']['fit_bias']=='offset':
            drivers+=[tf.expand_dims(tf.constant(offsets,dtype=tf.float32),0)]
        
        ### aggregation is done using addition here, could be extended to more complex differentiable functions
        if (self.dist[0]=='binomial') & (mode=='bam'):
            pred=tfkl.Activation('sigmoid')(abio_resp)*tfkl.Activation('sigmoid')(bio_resp)
            self.eta_model=None            
        else:
            logits=tfkl.Add(name=self.model_name+'_out')(drivers)   
            
            if self.dist[0] in ['negbin']:
                self.disp=GlobalParam(self.m,'disp')
                logits=self.disp(logits)
                self.nbr=NB(theta_var=self.disp.kernel)
                
            pred=tfkl.Activation(act_fn.get(self.dist[0]))(logits)
            self.eta_model=tfk.Model(self.inputs,logits) 
            
        self.pred_model=tfk.Model(self.inputs,pred)
        
        if self.var_assoc:
            self.assoc_model=tfk.Model(self.bio_in,assoc)
    
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
        
    
    def compile_model(self,on_logit=False,opt='adamax',mets=[]):
        self.on_logit=on_logit
        
        if self.dist[1] is not None:
            loss=loss_fn.get(self.dist[0])(self.dist[1])
            
        elif self.dist[0] in ['negbin']:
            loss=self.nbr.loss
        else:
            loss=loss_fn.get(self.dist[0])
            
        if on_logit:
            self.eta_model.compile(optimizer=opt,loss=loss,metrics=mets)
            
        else:
            self.pred_model.compile(optimizer=opt,loss=loss,metrics=mets)
            
    
    def fit_model(self,trainset,validset,train_config,vb,cbk=[]):
        # cbk+=[tfk.callbacks.EarlyStopping(
        #     monitor=train_config['objective'],min_delta=train_config['tol'],patience=train_config['patience'],mode=train_config['mode'])]
        
        self.pred_model.fit(x=[trainset['x'],trainset['y']],y=trainset['y'],
                            validation_data=validset,
                            batch_size=train_config['bsize'],epochs=train_config['max_iter'],
                            callbacks=cbk,verbose=vb)
        
    
    def evaluate_model(self,trainset):
        perfs=self.pred_model.evaluate(x=[trainset['x'],trainset['y']],y=trainset['y'])
        
        ## Get loglikelihood 
        ll=(perfs[0]-self.pred_model.losses[1]).numpy()
        
        ## Compute information criterion 
        k=self.get_parameter_size()
        n=trainset['y'].shape[0]*trainset['y'].shape[1]
        
        ## AIC
        aic=2*ll + 2*k
        aicc=aic + (2*k*k+2*k)/(n-k-1)
        bic=2*ll + k*np.log(n)
        
        mets=['obj']+self.pred_model.metrics_names
        perfs={mets[k]:perfs[k] for k in range(len(perfs))}
        perfs['loss']=ll
        perfs['aic']=aic
        perfs['aicc']=aicc
        perfs['bic']=bic
        return perfs
    
    def get_parameter_size(self):
        d=compute_effective_dimension(self.get_embeddings()['response'],self.get_embeddings()['effect'])
        w = self.fe_env_net.count_params()
        
        return d*self.m + w
    
        
    def get_embeddings(self):
        eff=self.association.eff_emb.numpy()
        resp=self.association.resp_emb.numpy()
        
        return {"response":resp,"effect":eff}
    
    def get_network(self):
        return np.dot(self.association.resp_emb.numpy(),self.association.eff_emb.numpy().T)*(1-np.eye(self.m)) 
    
    def get_abiotic_response(self):
        return self.fe_env_net.get_weights()
    
    def get_biotic_loadings(self):
        return self.fe_bio_net.get_weights()
    
    def predict_network(self,X_bio):
        if self.var_assoc:
            return self.assoc_model(X_bio)
        else:
            return self.get_network()
        
    
    def save_model(self,file):
        self.pred_model.save_weights(file)
        
        
        
        
