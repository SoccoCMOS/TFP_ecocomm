# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 15:59:42 2020

@author: saras
"""


from lego_blocks import *
from eval_functions import *
from NB import NB
import numpy as np
from scipy.special import logit

 
seed=1234
tf.random.set_seed(seed)


def compute_density(mat,th=0.01):
    return np.sum(np.abs(mat)>=th)/(mat.shape[0]*mat.shape[1])

def compute_effective_dimension(resp,eff,eps=1E-4):
    rdimuse=(np.max(np.abs(resp),axis=0)>eps)
    edimuse=(np.max(np.abs(eff),axis=0)>eps)
    return np.sum(rdimuse | edimuse)
                

class TaxaEmbedding(tfkl.Layer):
    def __init__(self, output_dim,name,norm,groups, **kwargs):
       super(TaxaEmbedding, self).__init__(**kwargs)
       self.output_dim = output_dim
       self.vname=name
       self.norm=norm
       ## The following sets the species groups for embedding sharing
       ## If provided by the user these are used otherwise each species is considered
       ## As a sole group 
       self.groups=groups
       self.ng=len(np.unique(self.groups))

    def build(self, input_shapes):
       if self.norm=='unit':
           print('Adding unit norm constraints')
           emb_const=tfk.constraints.UnitNorm(axis=1)
       elif self.norm=='nneg':
           print('Adding non negative constraints')
           emb_const=tfk.constraints.NonNeg()
       else:
           emb_const=None
       self.kernel = self.add_weight(name=self.vname, 
                                     shape=[self.ng,self.output_dim[1]], 
                                     initializer='uniform', 
                                     #regularizer=l1,
                                     constraint= emb_const,
                                     trainable=True)       
       
       super(TaxaEmbedding, self).build(input_shapes)  

    def call(self, inputs):
       return self.kernel
       
    def compute_output_shape(self):
       return self.output_dim
    
class AssociationEmbedding(tfkl.Layer):
    def __init__(self, output_dim=None,sym=True,reg=None,mask=True,groups=None, **kwargs):       
       ## Get dimension
       self.output_dim = output_dim
       self.is_symmetric=sym
       ## Prepare mask for diagonals
       self.mask=tf.ones((self.output_dim,self.output_dim))
       if mask:
           self.mask-=tf.eye(self.output_dim)
           
       if (reg[0]>0) & (reg[1]>0):
           self.assoc_reg=tfkr.l1_l2(l1=reg[0],l2=reg[1])
       else:
           self.assoc_reg=None
       
       self.reg_emb=reg[2] if len(reg)>2 else True
       self.norm=reg[3] if len(reg)>3 else None 
       
       self.groups=np.arange(output_dim) if groups is None else np.array(groups)
       self.prior=groups is not None

       super(AssociationEmbedding, self).__init__(**kwargs)

    def build(self, input_shape):
       ## Parameters are embeddings of response and effect
       emb_dim=tf.constant([self.output_dim,input_shape[2]])  ## m  x  d
       if self.is_symmetric:
           self.eff_emb=self.resp_emb=TaxaEmbedding(emb_dim,'latent',self.norm,self.groups)(None)
       else:
           self.eff_emb=TaxaEmbedding(emb_dim,'effect',self.norm,self.groups)(None)
           self.resp_emb=TaxaEmbedding(emb_dim,'response',self.norm,self.groups)(None)
       
       ### Add regularization constraints over associations
       ## Sparsity inducing constraint
       if self.reg_emb:
           self.add_loss(lambda: self.assoc_reg(self.resp_emb)+self.assoc_reg(self.eff_emb))
           
       elif self.assoc_reg is not None:
           self.add_loss(lambda: self.assoc_reg(tf.tensordot(self.resp_emb,self.eff_emb,axes=[1,1])))
       
       super(AssociationEmbedding, self).build(input_shape)  

    def call(self, loadings):  
        ### inputs has dimension [n,1,d] for shared loadings or [n,m,d] for specific loadings per species
        ## alpha=[m,d], input=[n,1,d]   => [n,m,d]
        if self.prior:
            alpha=tf.gather(params=self.eff_emb,indices=self.groups,axis=0)
            rho=tf.gather(params=self.resp_emb,indices=self.groups,axis=0)
        else:
            alpha=self.eff_emb
            rho=self.resp_emb
            
        loaded_effect=tf.multiply(loadings,alpha)
        
        ## associations : [n,m,d] x [m,d] => [n,m,m]
        local_assoc=tfk.backend.dot(loaded_effect,tf.transpose(rho))

        return tf.multiply(local_assoc,self.mask)

    def compute_output_shape(self,input_shape):
       return tf.constant([input_shape[0],self.output_dim,self.output_dim])


class EcoAssocNet(object):
    def __init__(self,model_name="model",groups=None,shared_config=None,hsm_config=None,im_config=None):
        self.model_name=model_name
        self.hsm_config=hsm_config
        self.im_config=im_config
        self.shared_config=shared_config
        
        self.d=self.shared_config['latent_dim']
        self.m=self.shared_config['pool_size']
        self.var_assoc=(len(self.im_config['input'])>0)
        
        self.dist=shared_config['dist']
        self.groups=groups
        
    def create_architecture(self,offsets=None,mode='add',in_proba=False):
        ## generate abiotic and biotic covariates
        self.gen_covs()
        
        ## Abiotic response
        x_abio=self.fe_env_net(self.env_feat)
        
        ## Here, replace following dense layer by another network that yields regression parameters given traits
        if in_proba:
            print('HSM proba given')
            abio_resp=x_abio
        else:
            abio_resp=tfkl.Dense(self.m,use_bias=self.hsm_config['archi']['fit_bias'])(x_abio)
        
        ## Generate biotic response
        if self.var_assoc:
            x_bio=tf.expand_dims(self.fe_bio_net(self.bio_feat),axis=1)
        else:
            x_bio=tfkl.Lambda(lambda x: tf.ones((tf.shape(x)[0],1,self.d)),name=self.model_name+'_loadings')(abio_resp)
        
        assoc=self.association(x_bio)        
        bio_resp=tfkl.Dot(axes=1)([self.counts,assoc])
        
        ### Aggregate abiotic and biotic effects
        drivers=[abio_resp,bio_resp]
        # if self.im_config['archi']['fit_bias']=='offset':
        #     drivers+=[tf.expand_dims(tf.constant(offsets,dtype=tf.float32),0)]
        
        ### aggregation is done using addition here, could be extended to more complex differentiable functions
        if (self.dist[0]=='binomial') & (mode=='bam'):
            ## Add an intercept (useful for basal species)
            off_bio_resp=BiasLayer(name=self.model_name+'_indep_offset')(bio_resp)
            pred=tfkl.Activation('sigmoid')(abio_resp)*tfkl.Activation('sigmoid')(off_bio_resp)
            self.eta_model=tfk.Model(self.inputs,[abio_resp,off_bio_resp])            
        else:
            logits=tfkl.Add(name=self.model_name+'_out')(drivers)   
            
            if self.dist[0] in ['negbin']:
                self.disp=GlobalParam(self.m,'disp')
                logits=self.disp(logits)
                self.nbr=NB(theta_var=self.disp.kernel)
                
            pred=tfkl.Activation(act_fn.get(self.dist[0]))(logits)
            self.eta_model=tfk.Model(self.inputs,logits) 
            
        self.pred_model=tfk.Model(self.inputs,pred)
        self.hsm_model=tfk.Model(self.env_in,abio_resp)
        
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
            
        self.association=AssociationEmbedding(self.m,self.im_config['sym'],self.shared_config['reg'],True,self.groups)
        
    
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
                            validation_split=0.1,
                            callbacks=cbk,verbose=vb)
        
    def set_hsm_weights(self,init_hsm=None):
        init_=[init_hsm[:,1:].T,init_hsm[:,0]]
        self.hsm_model.set_weights(init_)
        
        
    def predict(self,X,Yc=None):  
        ##predicts given other species are absent in case of binary classif
        ##or given mean abundances of other species in case of count 
        if Yc is None:
            Yc=np.zeros((X.shape[0],self.m))
        y_pred=self.pred_model.predict(x=[X,Yc])
        
        return y_pred
        
    def evaluate_model(self,trainset):
        perfs=self.pred_model.evaluate(x=[trainset['x'],trainset['y']],y=trainset['y'])
        
        ## Get loglikelihood 
        ll=perfs[0]
        if len(self.pred_model.losses)>1:
            ll=(ll-self.pred_model.losses[1]).numpy()
        
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
        resp=self.association.resp_emb.numpy()
        eff=self.association.eff_emb.numpy()
        # if self.groups is not None:
        #     rho=resp[self.groups,:]
        #     alpha=eff[self.groups,:]
        
        return np.dot(resp,eff.T)*(1-np.eye(resp.shape[0])) 
        
    
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
        
        
        
        
