# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 10:31:31 2020

@author: saras

Adapted from Di Chen et al 2018 (E2EL for the DeepMVP)
"""

from lego_blocks import *
from eval_functions import *
import numpy as np
from scipy.special import expit


seed=1234
tf.random.set_seed(seed)

###############################################################################################################
'''
                                                ARCHITECTURE 
'''
###############################################################################################################
'''
Experimental
'''

VALIDATE_ARGS = True

class MVNPrecisionCholesky(tfd.TransformedDistribution):
  """Multivariate normal parameterized by loc and Cholesky precision matrix."""
  """ From Tensorflow probability tutorials """

  def __init__(self, loc, precision_cholesky, name=None):
    super(MVNPrecisionCholesky, self).__init__(
        distribution=tfd.Independent(
            tfd.Normal(loc=tf.zeros_like(loc),
                       scale=tf.ones_like(loc)),
            reinterpreted_batch_ndims=1),
        bijector=tfb.Chain([
            tfb.Affine(shift=loc),
            tfb.Invert(tfb.Affine(scale_tril=precision_cholesky,
                                  adjoint=True)),
        ]),
        name=name)
    
class Noise_chol(tfkl.Layer):
    '''
    This custom layer is parameterized with the (low rank) residual covariance matrix.
    Parameters: r_sqrt_sigma
    Hyperparameters: number of samples, rank of residual matrix 
    Input: mu of shape (None, number of tasks)
    Output: Wijk for ith observation, jth task, kth sample  
    '''
    def __init__(self, ntasks=16, nsamples=1000,regularizer={'l1':0.,'l2':0.},name='noise_chol'):
        ### Here we create the object once and for all, but the sampling is done on the call because we will use different
        ### noise samples each time we go through this layer
        super(Noise_chol, self).__init__()
        self.ntasks=ntasks
        self.nsamples=nsamples
        self.lyname=name
        self.l1=regularizer['l1'] if regularizer['l1'] is not None else 0
        self.l2=regularizer['l2'] if regularizer['l2'] is not None else 0
                
    
    def build(self,input_shapes):
        # self.noise_layer=tfpl.DistributionLambda(
        #     make_distribution_fn=lambda x: tfd.Normal(loc=tf.zeros((tf.shape(x)[0],self.ntasks,self.nsamples)),
        #                                               scale=tf.ones((tf.shape(x)[0],self.ntasks,self.nsamples))),
        #                                               name=self.lyname+'_zsmp')
        
        #### Define upper triangular cholesky factor of the precision matrix
        ##Initializer glorot uniform in=m out=(m-1)/2 
        u_omega_init=tf.random_uniform_initializer(
            minval=-np.sqrt(12.0/(3*self.ntasks-1)), 
            maxval=np.sqrt(12.0/(3*self.ntasks-1))
            )
        
        d_omega_init=tf.random_uniform_initializer(
            minval=-np.sqrt(6.0/self.ntasks), 
            maxval=np.sqrt(6.0/self.ntasks)
            )        
        
        #Fill only off-diagonal elements
        low_mask=tf.constant(np.triu(np.ones((self.ntasks,self.ntasks)), 1),dtype=tf.float32)
        
        self.u_omega = tf.Variable(u_omega_init(shape=[self.ntasks,self.ntasks],dtype=tf.float32)*low_mask,
                                   trainable=True,name=self.lyname+'_uomega')
        
        self.d_omega = tf.Variable(d_omega_init(shape=[self.ntasks],dtype=tf.float32),
                                   trainable=True,name=self.lyname+'_domega')

        if (self.l1>0)|(self.l2>0):
            reg_loss=tfkr.l1_l2(self.l1,self.l2)
            print('Sparsification penalty on cholesky factor')
            self.add_loss(lambda: reg_loss(self.u_omega))        
        
        super(Noise_chol, self).build(input_shapes)  
        
    def call(self, inputs):
        #noise=self.noise_layer(inputs)
        noise=tf.random.normal((tf.shape(inputs)[0],self.ntasks,self.nsamples),0.,1.,tf.float32,
                                name=self.lyname+'_zsmp')
        
        triu_omega=tf.linalg.set_diag(self.u_omega,tf.exp(self.d_omega))
        #triu_omega=self.u_omega+tf.exp(self.d_omega)*tf.eye(self.ntasks)
        
        x = tf.map_fn(lambda rhs: tf.linalg.triangular_solve(triu_omega,rhs,lower=False), 
                                noise, dtype=tf.float32)
        #Solve  UX=Z   Z:(n,m,ns) U:(m,m) X:(n,m,ns)
        r_sample=tf.transpose(x,perm=[0,2,1])+tf.expand_dims(inputs,axis=1) ##(n,m,ns)
        return r_sample

#####################################################################################################
class Noise(tfkl.Layer):
    '''
    This custom layer is parameterized with the (low rank) residual covariance matrix.
    Parameters: r_sqrt_sigma
    Hyperparameters: number of samples, rank of residual matrix 
    Input: mu of shape (None, number of tasks)
    Output: Wijk for ith observation, jth task, kth sample  
    '''
    def __init__(self, rank=4, ntasks=16, nsamples=1000,regularizer={'target':'cov','l1':0.,'l2':0.},name='noise'):
        ### Here we create the object once and for all, but the sampling is done on the call because we will use different
        ### noise samples each time we go through this layer
        super(Noise, self).__init__()
        self.rank=rank
        self.ntasks=ntasks
        self.nsamples=nsamples
        self.lyname=name
        
        ### Here we just define and initialize the parameters according to input_dim and output_dim (units)
        ## sqsigma = ntasks x rank (rank<= ntasks)
        init_sqsigma=tf.random_uniform_initializer(minval=-np.sqrt(6.0/(ntasks+rank)), maxval=np.sqrt(6.0/(ntasks+rank)))
        
        self.sqsigma=tf.Variable(init_sqsigma(shape=(rank,ntasks)),
                                 dtype=tf.float32,trainable=True,name=self.lyname+'_sqsigma')
        
        ### Same the noise sampling layer should be created once and then called for different batches on the call function
        # self.noise_layer=tfpl.DistributionLambda(
        #     make_distribution_fn=lambda x: tfd.Normal(loc=tf.zeros((tf.shape(x)[0],self.nsamples,self.rank)),scale=tf.ones((tf.shape(x)[0],self.nsamples,self.rank))),name=self.lyname+'_zsmp') 
        
        self._reg_cov(regularizer)

    def call(self, inputs):
        #noise=self.noise_layer(inputs)
        noise=tf.random.normal((tf.shape(inputs)[0],self.nsamples,self.rank),0.,1.,tf.float32,
                               name=self.lyname+'_zsmp')
        
        r_sample=tf.matmul(noise,self.sqsigma)+tf.expand_dims(inputs,axis=1)
        return r_sample
    
    def _reg_cov(self,reg_params):     ###Regularize on covariance or precision matrix (after inversion) 
        ### Add norm constraint on covariance/precision matrix
        if reg_params is not None:
            self.l1=reg_params['l1'] if reg_params['l1'] is not None else 0
            self.l2=reg_params['l2'] if reg_params['l2'] is not None else 0
            
            if (self.l1>0) | (self.l2>0):
                reg_loss=tfkr.l1_l2(self.l1,self.l2)
                self.add_loss(lambda : reg_loss(self.sqsigma))
  

  
class DeepMVP(object):
    def __init__(self,model_name='dmvp',in_config=None,fe_config=None,out_config=None,
                 m=1,r=3,nsamples=100,agg_samples=True,cov=True,link='probit',
                 ):
        self.model_name=model_name
        #self.dmvp=None
        
        ### Architecture configuration ###
        self.in_config=in_config
        self.fe_config=fe_config
        self.out_config=out_config
        
        self.activation='sigmoid' if link=='logit' else 'linear' if link=='linear' else (lambda x: 
                              tfd.Normal(loc=0.,scale=1.).cdf(x)*(1-self.eps1)+self.eps1*0.5)
        
        self.ntasks=m
        self.rank=r
        self.nsamples=nsamples
        self.cov=cov
        
        self.agg_samples=agg_samples
        
        ### 
        self.eps2=tf.constant(1e-6*2.0**(-100), dtype="float64")
        self.eps1=tf.constant(1e-6, dtype="float32")
        self.eps3=1e-30
        
        
    def create_architecture(self):
        '''
        1) Getting the mean
        '''
        
        ###Input features encoding ###
        l_in_E,l_E=custom_input(self.in_config,concat_feat=True,model_name='in_fe_'+self.model_name)
        
        ###Feature extraction component ###
        self.feat_net=mtl_output(self.fe_config,self.out_config,model_name='mu_'+self.model_name,path='train/',plot=False,concat=False)
        
        ### Mean mu prediction => simple multispecies model ###
        r_mu=self.feat_net(l_E)
        
        c_r_mu=tfkl.Concatenate(axis=1)(r_mu)
        
        '''
        2) Generating noise for the residual distribution
        '''
        if self.rank>0:
            if self.cov:
                self.noise_layer=Noise(rank=self.rank, ntasks=self.ntasks, nsamples=self.nsamples,regularizer=self.fe_config['assoc_pen'],name=self.model_name+'_noise')
            else:
                self.noise_layer=Noise_chol(ntasks=self.ntasks, nsamples=self.nsamples,regularizer=self.fe_config['assoc_pen'],name=self.model_name+'_noisechol')
            r_sample=self.noise_layer(c_r_mu)
            
            '''
            3) Generating output predictions
            '''
            ### Probit activation
            Y_hat=tfkl.Activation(activation=self.activation)(r_sample)

            if self.agg_samples:
                 Y_hat=tfkl.Lambda(lambda x: tf.reduce_mean(x,axis=1),name=self.model_name+'_aggsamples')(Y_hat)
        
        else: 
            Y_hat=tfkl.Activation(activation=self.activation)(c_r_mu)
   
        self.dmvp=tfk.Model(l_in_E,Y_hat,name=self.model_name)#
        
        self.mhsm=tfk.Model(l_in_E,c_r_mu)
        
    def plot_model(self,file):
        tfkv.plot_model(self.dmvp,file,show_shapes=True)
        
    
    def compile_dmvp(self,opt='adam',loss='binary_crossentropy',metrics=['binary_accuracy'],gpus=1):
        print('Optimizer %s' % opt)
        self.dmvp.compile(optimizer=opt,loss=loss,metrics=metrics)
        
    def fit_dmvp(self,trainset):
        self.dmvp.fit(x=trainset['X'],y=trainset['y'],validation_split=0.1)
        
        
    def predict(self,X_test):
        y_pred=self.dmvp.predict(X_test)
        if self.rank<-1:
            y_pred=expit(y_pred)
            
        return y_pred
    
    def evaluate(self,X_test,y_test,metrics):
        y_pred=self.predict(X_test)
        
        if len(y_pred.shape)==3:
            y_pred_=tf.constant(np.mean(y_pred,axis=1))
        else:
            y_pred_=y_pred
            
        y_true=tf.constant(y_test)
        
        return {f:fct_metrics.get(f)(y_true,y_pred_).numpy() for f in metrics}
    
    def inspect_model(self):
        fe_weights=self.feat_net.get_weights()

        ## Residual covariance (latent factors)
        if self.cov:
            latent_factors=self.noise_layer.sqsigma.numpy()
        
        else:
            latent_factors=self.noise_layer.u_omega.numpy()
            np.fill_diagonal(latent_factors,
                            np.exp(self.noise_layer.d_omega.numpy())
                            )
        
        res_cov=np.dot(latent_factors.T ,latent_factors)
        
        return {'feat_net':fe_weights,'latent':latent_factors,'cov':res_cov}
                


###############################################################################################################
'''
                                                Evaluation
'''
###############################################################################################################
#### Loss from DMVP paper ####

def nll_loss(y_true,y_pred):    
        y_true_=tf.cast(tf.expand_dims(y_true,axis=1),dtype=tf.float32)
        y_pred_=tf.convert_to_tensor(y_pred)
        
        ### Use my WBCE implementation 
        sample_nll=tfk.backend.binary_crossentropy(y_true_,y_pred_)
        #sample_nll=bce(bw=class_weight,lw=label_weight,avg=False)(y_true_,y_pred_)
        
        logprob=-tf.reduce_sum(sample_nll, axis=2)
    		
    	#the following computation is designed to avoid the float overflow
        maxlogprob=tf.reduce_max(logprob, axis=1,keepdims=True)
        
        Eprob=tf.reduce_mean(tf.exp(logprob-maxlogprob), axis=1,keepdims=True)
        nll=tf.reduce_mean(-tfm.log(Eprob)-maxlogprob)
        
        return nll
    
def weighted_nll_loss(label_weight,class_weight):
    def nll_loss(y_true,y_pred):    
        y_true_=tf.cast(tf.expand_dims(y_true,axis=1),dtype=tf.float32)
        y_pred_=tf.convert_to_tensor(y_pred)
        
        ### Use effiscient and stable WBCE implementation 
        #sample_nll=tfk.backend.binary_crossentropy(y_true_,y_pred_)
        sample_nll=bce(bw=class_weight,lw=label_weight,avg=False)(y_true_,y_pred_)
        
        logprob=-tf.reduce_sum(sample_nll, axis=2)
    		
    	#the following computation is designed to avoid the float overflow
        maxlogprob=tf.reduce_max(logprob, axis=1,keepdims=True)
        
        Eprob=tf.reduce_mean(tf.exp(logprob-maxlogprob), axis=1,keepdims=True)
        nll=tf.reduce_mean(-tfm.log(Eprob)-maxlogprob)
        
        return nll
    
    return nll_loss	


fct_loss={
    'binary_crossentropy':tfk.losses.binary_crossentropy,
    'cw_bce':lambda x: bce(bw=x[1],lw=x[0],avg=True),
    'nll_loss':nll_loss,
    'fl':lambda x: stable_focal_loss(gamma=x[0],alpha=x[1]),
    'w_nll_loss':lambda x: weighted_nll_loss(label_weight=x[0], class_weight=x[1])
    }

fct_metrics={
    'binary_accuracy':tfkm.BinaryAccuracy(threshold=0.5),
    'auc':tfkm.AUC(curve='ROC',multi_label=True),
    'precision':tfkm.Precision(),
    'recall':tfkm.Recall(),
    'tp':tfkm.TruePositives(),
    'tn':tfkm.TrueNegatives(),
    'fp':tfkm.FalsePositives(),
    'fn':tfkm.FalseNegatives(),
    'pr':tfkm.AUC(curve='PR',multi_label=True)}


class EVal_epoch(tf.keras.callbacks.Callback):
  def __init__(self,period,dmvp_object,trainset,testset,evalset,metrics,metric_writer=None,specific_writers=None,th=0.5,prevs=None,names=None):
      self.period=period
      self.dmvp=dmvp_object
      
      if trainset is not None:
          self.X_train=trainset['X']
          self.y_train=trainset['y'].values
          self.logtrain=True
          
      else:
          self.logtrain=False
      
      if testset is not None:
          self.X_test=testset['X']
          self.y_test=testset['y'].values
          self.logtest=True
          
      else:
          self.logtest=False          
      
      if evalset is not None:
          self.X_eval=evalset['X']
          self.y_eval=evalset['y']  ### (rowid,gids) , gids=index in column 
          self.logeval=True
          self.max_score=-1
          self.best_model=None
          
      else:
          self.logeval=False          
          
      self.metrics=metrics
      self.th=th
      
      self.names=names
      self.prevs=prevs if prevs is not None else trainset['y'].mean(axis=0).tolist()
      
      self.train_perfs=None
      self.task_perfs=None
      self.test_perfs=None
      self.eval_perfs=[]
      
      
  def on_epoch_begin(self, epoch, logs=None):
    if (epoch%self.period)==0:
        
        #### Plot overall metrics ####
        ### Training set
        if self.logtrain:
            train_perfs=self.dmvp.evaluate(self.X_train,self.y_train,self.metrics)
            for f in self.metrics:
                tf.summary.scalar(f+'/train', data=train_perfs[f], step=epoch)
            
            train_perfs['sensitivity']=train_perfs['tp']/(train_perfs['tp']+train_perfs['fn'])
            train_perfs['specificity']=train_perfs['tn']/(train_perfs['tn']+train_perfs['fp'])
            train_perfs['tss']=train_perfs['sensitivity']+train_perfs['specificity']-1
            
            tf.summary.scalar('sensitivity/train', data=train_perfs['sensitivity'], step=epoch)
            tf.summary.scalar('specificity/train', data=train_perfs['specificity'], step=epoch)
            tf.summary.scalar('tss/train', data=train_perfs['tss'], step=epoch)
              
            self.train_perfs=train_perfs
            
        ### Test set
        if self.logtest:            
            test_perfs=self.dmvp.evaluate(self.X_test,self.y_test,self.metrics)
            for f in self.metrics:
                tf.summary.scalar(f+'/test', data=test_perfs[f], step=epoch)            
            
            test_perfs['sensitivity']=test_perfs['tp']/(test_perfs['tp']+test_perfs['fn'])
            test_perfs['specificity']=test_perfs['tn']/(test_perfs['tn']+test_perfs['fp']) 
            test_perfs['tss']=test_perfs['sensitivity']+test_perfs['specificity']-1
            
            tf.summary.scalar('sensitivity/test', data=test_perfs['sensitivity'], step=epoch)
            tf.summary.scalar('specificity/test', data=test_perfs['specificity'], step=epoch)
            tf.summary.scalar('tss/test', data=test_perfs['tss'], step=epoch)
            
            self.test_perfs=test_perfs
               
            #### Plot task-specific metrics ####
            test_pred=self.dmvp.predict(self.X_test)
            if len(test_pred.shape)==3:
                test_pred=np.mean(test_pred,axis=1)
                
            test_eval=eval_task(y_true=self.y_test, y_pred=test_pred,taxa=None,th=self.th,prevs=self.prevs)        
            for j, t in enumerate(test_eval):
                tf.summary.scalar('taskwise/'+self.names[j]+'/prev',data=t.get('prev'),step=epoch)
                for k in ['accuracy', 'recall', 'precision', 'f1', 'sensitivity', 'specificity', 'tss', 'roc_auc']:                
                    tf.summary.scalar('taskwise/'+self.names[j]+'/'+k,data=t.get(k),step=epoch)
            
            self.task_perfs=test_eval
        
        ### Eval set
        if self.logeval:
            y_pred_ev=self.dmvp.predict(self.X_eval)
            if len(y_pred_ev.shape)==3:
                y_pred_ev=np.mean(y_pred_ev,axis=1)
            
            probs=y_pred_ev[self.y_eval] ##index to retrieve predictions for specific observed taxa
            
            ### How to evaluate y_pred_ev  vs y_ev ? Given that it's an occurrence only
            ce_score=np.mean(-np.log(probs+eps))
            eval_class=(np.array(probs)>=self.th).astype(int)
            recall_score=np.mean(eval_class)
            
            if recall_score>self.max_score:
                self.max_score=recall_score
                self.best_model=self.dmvp.dmvp.get_weights()
            
            tf.summary.scalar('eval/recall',data=recall_score,step=epoch)
            tf.summary.scalar('eval/ce',data=ce_score,step=epoch)
            
            self.eval_perfs.append(recall_score)
            
            
            
        
        
                
