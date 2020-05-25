# -*- coding: utf-8 -*-
"""
Created on Sat May  2 11:38:16 2020

@author: saras
"""


from deep_mvp import *
import numpy as np
import scipy.linalg as lin
import scipy.special as sp
import pandas as pd
import scipy
import sklearn.metrics as skm
from sklearn.datasets import make_sparse_spd_matrix


n=100
m=2
p=1
ns=1000
r=2


####################################################################################################
def simulate_dist(n,m,p,r):
    ## precision
    prng = np.random.RandomState(1)
    
    ### Simulate parameters ###
    if p>0:
        beta=np.random.normal(np.zeros(p),np.ones(p),(m,p))
        biases=np.random.uniform(size=m)
        
        ### Simulate data ###
        x=np.random.normal(np.zeros(p),np.ones(p),(n,p))
        
        ### Simulate responses
        mu=np.dot(x,beta.T)+biases
        
    else:
        mu=np.zeros((n,m))
        beta=biases=None
        
    prec = make_sparse_spd_matrix(m, alpha=.75,
                              smallest_coef=.2,
                              largest_coef=.7,
                              random_state=prng)
    
    cov=np.linalg.inv(prec)
        
    probit=mu+np.random.multivariate_normal(np.zeros(m),cov,n)      

    y=(probit>0).astype(int)
    
    return (x,y,beta,biases,cov,prec)

####################################################################################################
### Inputs ###
in_feat=[('num',{'id':'env','dim':p})]

### Feature extraction ###
fe_config={'name':'fe','archi':{'nbnum':p,'nl':0,'nn':[],'activation':None},
           'reg':None,'assoc_pen':{'target':'prec','l1':0.,'l2':0.}}

### Setting up specific configuration for each taxa
taxa={'name':'taxa','archi':{'fit_bias':True,'nbnum':p,'nl':1,'nn':[1],'activation':['linear']},
      'reg':None}

out_config=[ {'name':'mean_%d'%j,'type':'continuous','specific':taxa,'activation':'linear'} for j in range(m)]

### Model name
mod_name='mvp_sim'

#### Creating architectures ####
dmvp_obj=DeepMVP(mod_name,in_feat,fe_config,out_config,m=m,
             r=r,nsamples=ns,agg_samples=False,cov=True)

dmvp_obj.create_architecture()

lf=fct_loss.get('nll_loss')
dmvp_obj.compile_dmvp(loss=lf,metrics=[],gpus=1,opt='adam')

tfkv.plot_model(dmvp_obj.dmvp,show_shapes=True)
####################################################################################################     

x,y_sim,true_beta,true_bias,true_cov,true_prec=simulate_dist(n,m,p,r)
true_chol=np.linalg.cholesky(true_prec)


true_mean = np.zeros([2], dtype=np.float32)
# We'll make the 2 coordinates correlated
true_cor = np.array([[1.0, 0.9], [0.9, 1.0]], dtype=np.float32)
# And we'll give the 2 coordinates different variances
true_var = np.array([4.0, 1.0], dtype=np.float32)
# Combine the variances and correlations into a covariance matrix
true_cov = np.expand_dims(np.sqrt(true_var), axis=1).dot(
    np.expand_dims(np.sqrt(true_var), axis=1).T) * true_cor
# We'll be working with precision matrices, so we'll go ahead and compute the
# true precision matrix here
true_precision = np.linalg.inv(true_cov)

chol=np.linalg.cholesky(true_cov)

x=np.random.normal(0,1,(n,1))
beta=[1,1]
y_sim=(beta*x+np.random.multivariate_normal(true_mean,true_cov,n)>0).astype(int)

import seaborn as sns
sns.scatterplot(data=pd.DataFrame(y_sim,columns=['y1','y2']),x='y1',y='y2')

####################################################################################################

### First, we set the true parameters, and we check that it predicts the true data ###
init_weights=dmvp_obj.dmvp.get_weights()

params=init_weights.copy()

for j in range(m):
    params[2*j]=true_beta[j:j+1,:].T
    params[2*j+1]=np.array([true_bias[j]])


params[m*2]=chol#*(1-np.eye(m))
#params[m*2+1]=np.log(np.diag(true_chol))


dmvp_obj.dmvp.set_weights(init_weights)
y=dmvp_obj.predict(x).mean(axis=1)#>0.5).astype(int)
skm.accuracy_score(y_sim,(y>0.5).astype(int))
skm.roc_auc_score(y_sim, y,'macro')

dmvp_obj.dmvp.set_weights(params)
y=dmvp_obj.predict(x).mean(axis=1)#>0.5).astype(int)
skm.accuracy_score(y_sim,(y>0.5).astype(int))
skm.roc_auc_score(y_sim, y,'macro')

dmvp_obj.dmvp.set_weights(init_weights)
dmvp_obj.dmvp.fit(x=x,y=y_sim,batch_size=64,epochs=100)

### Predict
#mu_pred=dmvp_obj.mhsm.predict(x)
y_pred=dmvp_obj.predict(x).mean(axis=1)
skm.roc_auc_score(y, y_pred,'macro')

theta=dmvp_obj.dmvp.get_weights()[20]+np.diag(np.exp(dmvp_obj.dmvp.get_weights()[21]))

omega=np.dot(theta.T,theta)

import seaborn as sns
sns.heatmap(omega+np.eye(m),cmap='seismic')
sns.heatmap(true_prec,cmap='seismic')

### Second, we train from scratch and evaluate wrt true parameters           