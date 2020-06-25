# -*- coding: utf-8 -*-
"""
Created on Mon May 25 18:04:32 2020

@author: saras
"""


'''
Select best combination of hyperparameters based on: 
    - CV  ~ best metrics
    - AIC, BIC, eBIC
    - Bootstrap + CI
    - StARS
'''

import timeit
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from ecoassocnet import *
import random
import json
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
skl=tfk.wrappers.scikit_learn
from scipy.special import expit

####
'''
Arguments of create_model are keys in param_grid
The following generates a grid of hyperparameters for the EA model, and it returns a method
for instantiating an estimator object given a configuration of hyperparameters
'''
####

def generate_grid(dist,p,m,c=0,sym=True,d_range=None,l_range_hsm=[0.,0.001,0.01,0.1],l_range_ass=[0.,0.001,0.01,0.1],l_range_im=[0.,0.001,0.01,0.1],
                  l_reg=[False],l_norm=[None],out_mode='add',groups=None,in_proba=False,
                  bs=64,max_it=200,tol=1E-6,patience=5,mode='min',obj='loss'):
    pb='classification' if dist[0] in ['binomial','binomial2'] else 'mclassification' if dist[0] in ['categorical'] else 'regression' if dist[0] in ['normal'] else 'count'
    
    param_combi={
        'd':np.arange(2,m) if d_range is None else d_range,
        'pen':[(l,0.,reg,norm) for l in l_range_ass for reg in l_reg for norm in l_norm],
        'opt':['adam'],
        'fit_bias_hsm':[True],'act_hsm':['relu'],'nn_hsm':[[m]],'reg_hsm':[l for l in l_range_hsm],
        'fit_bias_im':[False],'act_im':['relu'],'nn_im':[[]],'reg_im':[l for l in l_range_im],
        }
    
    param_grid=[]
    for d, pen, opt, fit_bias_hsm, act_hsm, nn_hsm, reg_hsm, fit_bias_im,act_im,nn_im,reg_im in itertools.product(
            np.arange(2,m) if d_range is None else d_range,
            [(l,0.,reg,norm) for l in l_range_ass for reg in l_reg for norm in l_norm],
            ['adam'],
            [True],
            ['relu'],
            [[]],
            [l for l in l_range_hsm],
            [False],
            ['relu'],
            [[]],
            [l for l in l_range_im]):
        conf={
        'd':d,
        'pen':pen,
        'opt':opt,
        'fit_bias_hsm':fit_bias_hsm,'act_hsm':act_hsm,'nn_hsm':nn_hsm,
        'reg_hsm':reg_hsm,'fit_bias_im':fit_bias_im,
        'act_im':act_im,'nn_im':nn_im,'reg_im':reg_im
        }
        
        param_grid.append(conf)
    
    def create_model(name="model",d=2,pen=(0.,0.0),opt='adamax',
                     fit_bias_hsm=True,act_hsm='relu',nn_hsm=[m],reg_hsm=(0.,0.),
                     fit_bias_im='intercept',act_im='relu',nn_im=[],reg_im=(0.,0.)
                     ):   
        
        
        regwh={'regtype':'l1_l2','regparam':reg_hsm,'dropout':None}
        regwi={'regtype':'l1_l2','regparam':reg_im,'dropout':None}
        
        shared_config={'latent_dim':d,'pool_size':m,'reg':pen,'dist':dist}
        hsm_config={'input':[('num',{'id':'num1','dim':p})],'reg':reg_hsm,'archi':{'nbnum':p,'nl':len(nn_hsm),'nn':nn_hsm,'activation':act_hsm,'fit_bias':fit_bias_hsm}}
        im_config={'input':[],'sym':sym,'reg':reg_im,
                    'archi':{'nbnum':c,'nl':len(nn_im),'nn':nn_im,'activation':act_im,'fit_bias':fit_bias_im}}
        
        
        train_config={'bsize':bs,'max_iter':max_it,
                      'tol':tol,'patience':patience,
                      'mode':mode,'objective':obj}
        
        ea=EcoAssocNet(model_name=name,groups=groups,shared_config=shared_config,hsm_config=hsm_config,im_config=im_config)
        ea.create_architecture(mode=out_mode,in_proba=in_proba) 
        ea.compile_model(opt=opt,mets=metric_fn.get(pb))
        return ea, train_config
    
    return create_model, param_grid

######################################

##########################################

class EAModelSelection(object):
    def __init__(self,data,dist,sym,classif,mode='add',name='msel',groups=None,in_proba=False,bootstrap=1,rs=1):
        self.hsm_covariates=data[0]
        self.targets=data[1]
        self.im_covariates=data[2]
        
        self.n, self.m=self.targets.shape
        self.p=self.hsm_covariates.shape[1]
        self.c=0 if self.im_covariates is None else self.im_covariates.shape[1]
        
        self.sym=sym
        self.dist=dist
        self.mode=mode
        self.groups=groups
        self.in_proba=in_proba
        
        if bootstrap>1:
            self.bootstrap=self.bootstrap_dataset(name=name,nsamples=bootstrap,rs=rs)
        
        ## Replace the following wrapper with custom estimator
        #self.estimator=EcoAssocNetClassifier if classif else EcoAssocNetRegressor
    
    def generate_grid(self,d_range=None,l_reg=[False],l_norm=[None],l_range_hsm=None,l_range_im=None,l_range_ass=None,max_it=10):
        if l_range_ass is None:
            l_range_ass=[0.,0.001,0.01,0.1]
            
        if l_range_hsm is None:
            l_range_hsm=[0.,0.001,0.01,0.1] 
            
        if l_range_im is None:
            l_range_im=[0.,0.001,0.01,0.1]      
        
        self.l_range_ass=l_range_ass
        self.l_range_hsm=l_range_hsm
        self.l_range_im=l_range_im
        
        self.l_reg=l_reg
        self.l_norm=l_norm
        
        self.create_model, self.param_grid=generate_grid(self.dist,self.p,self.m,self.c,out_mode=self.mode,in_proba=self.in_proba,groups=self.groups,sym=self.sym,l_reg=self.l_reg,l_norm=self.l_norm,d_range=d_range,l_range_hsm=self.l_range_hsm,l_range_im=self.l_range_im,l_range_ass=self.l_range_ass,max_it=max_it)
        self.path=None
    
    def information_model_selection(self,save=True,plot_network=False,plot_hsm=None,
                                    file='weights',init_weights=None,cbks=[]):
        scores=[]
  
        for cpt in range(len(self.param_grid)):
            conf=self.param_grid[cpt]
            ea_obj, train_config=self.create_model(**conf)
            
            if init_weights is not None:
                ea_obj.set_hsm_weights(init_weights)
            
            trainset={'x':self.hsm_covariates,'y':self.targets}
            ea_obj.fit_model(trainset,None,train_config,vb=1,cbk=cbks)
            perfs=ea_obj.evaluate_model(trainset)
            
            
            net=ea_obj.get_network()
            for k in perfs.keys():
                conf[k]=perfs.get(k)
                
            scores.append(conf)
            
            ## Save weights ##                        
            if save:
                ea_obj.save_model('%sweights_%d.h5'%(file,cpt))
            
            ## Plot network ##
            if plot_network:
                fig, ax=plt.subplots(1,1)
                sns.heatmap(net,cmap='seismic',vmin=-1,vmax=1,ax=ax)
                
                fig.savefig(
                    '%snetwork_%d.png'%(file,cpt),bbox_inches='tight')
                
                plt.close()
                
            if plot_hsm is not None:
                xdata=plot_hsm[0]
                xlab=plot_hsm[1]
                ytrue=plot_hsm[2]
                ylab=plot_hsm[3]
                ypred=ea_obj.predict(X=xdata,Yc=ytrue)
                yhsm=expit(ea_obj.hsm_model.predict(xdata))
                
                m=len(ylab)
                nrow, ncol=plot_hsm[4]
                fig, ax=plt.subplots(nrow,ncol,figsize=(20,20))
                nb=0
                for i in range(nrow):
                    for j in range(ncol):
                        sns.lineplot(x=xlab,y=ypred[:,nb],ax=ax[i,j])
                        sns.scatterplot(x=xlab,y=ytrue[:,nb],ax=ax[i,j])
                        nb+=1
                
                fig.savefig(plot_hsm[5]+'pred_'+str(cpt)+'.pdf',bbox_inches='tight')
                plt.close()
                
                fig, ax=plt.subplots(nrow,ncol,figsize=(20,20))
                nb=0
                for i in range(nrow):
                    for j in range(ncol):
                        sns.lineplot(x=xlab,y=yhsm[:,nb],ax=ax[i,j])
                        sns.scatterplot(x=xlab,y=ytrue[:,nb],ax=ax[i,j])
                        nb+=1
                
                fig.savefig(plot_hsm[5]+'hsm_'+str(cpt)+'.pdf',bbox_inches='tight')
                plt.close()
      
        return scores
            
    def stars_network(self,beta=0.05,eps=1E-4,th=1E-2,reg_emb=True,cpt=0,conf=None,N=10,norm_emb=True):
        if conf is None:
            sel_conf={k:self.param_grid.get(k)[cpt] for k in self.param_grid.keys()}
        else:
            sel_conf=conf
        
        ## Generate subsamples
        b=int(10*np.sqrt(self.n))

        S=[random.sample(np.arange(self.n).tolist(), b) for i in range(N)]
        
        models={}
        ## Generate lasso path
        for lmd in self.l_range_ass:
            conf_lmd=sel_conf.copy()
            conf_lmd['pen']=(lmd,0.,reg_emb,norm_emb)
            
            samples=[]
            lmd_theta=np.zeros((self.m,self.m))
            densities=[]
            dimensions=[]
            
            for j, subset in enumerate(S):
                trainset={'x':self.hsm_covariates[subset,:],
                          'y':self.targets[subset,:]
                              }
                
                ea_obj, train_config=self.create_model(**conf_lmd)
                
                ea_obj.fit_model(trainset,None,train_config,vb=1,cbk=[])
                net=ea_obj.get_network()
                resp=ea_obj.get_embeddings().get('response')
                eff=ea_obj.get_embeddings().get('effect')
                dens=compute_density(net,th)
                dim=compute_effective_dimension(resp,eff,eps)
                densities.append(dens)
                dimensions.append(dim)
                
                samples.append({'density':dens,
                               'effective_dimension':dim,
                               'associations':net,
                               'response':resp,
                               'effect':eff,
                               #'associations_n':net/np.diag(ea_obj.get_network()),
                               'weights':ea_obj.get_abiotic_response()
                               })
                
                del ea_obj
                
                lmd_theta+=(net>th).astype(int)
            
            ### Compute statistics over inferred networks on subsamples ###
            lmd_theta/=N
            lmd_var=2*lmd_theta*(1-lmd_theta)
            
            lmd_D=np.sum(lmd_var)/(self.m*(self.m-1))
            
            models[lmd]={'raw':samples,'Instability':lmd_D,'Density':densities,'Dimension':dimensions,'theta':lmd_theta}
        
        
        self.path=models
        ## Select lambda values such that D<=Beta, choose smallest lambda 
        self.lambda_sel=np.min([l for l in self.l_range_ass if models[l]['Instability']<=beta])
        
        return self.lambda_sel
    
    def plot_path(self):
        if self.path is None:
            print('StARS was not executed')
            
        else:
            ### Visualize path ###
            ## Plot along lambda path of D statistic, mean density, mean dimension 
            dens=pd.DataFrame(data=np.array([(l,s,x) for l in self.l_range_ass for s, x in enumerate(self.path.get(l)['Density'])]),
                              columns=['Penalty','Sample','Density'])
            
            dims=pd.DataFrame(data=np.array([(l,s,x) for l in self.l_range_ass for s, x in enumerate(self.path.get(l)['Dimension'])]),
                              columns=['Penalty','Sample','Dimension'])
            
            D=pd.DataFrame(data=np.array([(l,self.path.get(l)['Instability']) for l in self.l_range_ass]),
                           columns=['Penalty','Instability'])
            
            fig, ax=plt.subplots(3,1,figsize=(5,10))
            sns.boxplot(data=dens,x='Penalty',y='Density',ax=ax[0])
            sns.boxplot(data=dims,x='Penalty',y='Dimension',ax=ax[1])
            sns.lineplot(data=D,x='Penalty',y='Instability',ax=ax[2])
            
            ## Plot of association coefficients
            fig, ax=plt.subplots(1,len(self.l_range_ass),figsize=(10*len(self.l_range_ass),10))
            for i,l in enumerate(self.l_range_ass):
                sns.heatmap(self.path[l]['theta'],ax=ax[i],cmap='Greys',vmin=0,vmax=1)
                
            # coeffs=np.stack([np.ravel(self.path[l]['theta']) for l in self.l_range_ass],axis=0)
            
            # for i in range(coeffs.shape[0]):
            #     plt.plot(self.l_range_ass,coeffs[:,i])
            
            
            
    def grid_search_cv(self,cv=5,rep=1,vb=0,cbks=[],init_weights=None):
        
        ##
        # Selects best architecture based on best average performance on cross-validation 
        ##
        scores=[]
        for repi in range(rep):
            print('Repetition %d out of %d'%(repi,rep))
            folds=KFold(n_splits=cv).split(self.hsm_covariates,(self.targets>0).astype(int))
            f=0
            for idx_train, idx_test in folds:
                print('Fold %d out of %d'%(f,cv))
                f+=1
                trainset={'x':self.hsm_covariates[idx_train,:],'y':self.targets[idx_train,:]}
                testset={'x':self.hsm_covariates[idx_test,:],'y':self.targets[idx_test,:]}
                
                for cpt in range(len(self.param_grid)):
                    print('Configuration %d out of %d'%(cpt,len(self.param_grid)))
                    conf=self.param_grid[cpt].copy()
                    ea_obj, train_config=self.create_model(**conf)
                    
                    if init_weights is not None:
                        ea_obj.set_hsm_weights(init_weights)
                    
                    t0=timeit.timeit()
                    ea_obj.fit_model(trainset,None,train_config,vb=vb,cbk=cbks)
                    tf=timeit.timeit()
                    perfs=ea_obj.evaluate_model(testset)
                    
                    #net=ea_obj.get_network()
                    for k in perfs.keys():
                        conf[k]=perfs.get(k)
                    
                    conf['fold']=f
                    conf['repetition']=repi
                    conf['time']=tf-t0
                    scores.append(conf)  
        
        ### Select per criteria ###
                    
        return scores
    
    def bootstrap_dataset(self,name,nsamples=100,rs=1):
        X=self.hsm_covariates
        y=self.targets
        self.bootstrap=[]
        for bs in range(nsamples):
            ea=self.create_model('%s_%d'%(name,bs),**conf)
            boot = resample(np.arange(X.shape[0]), replace=True, n_samples=len(X), random_state=rs)
            X_train=X[boot,:]
            y_train=y[boot,:]
            
            perfs=ea.evaluate_model(X_train,y_train)
            self.bootstrap.append((ea,perfs)) 
        
        
    