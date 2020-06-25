# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 10:35:49 2020

@author: saras

This script contains specific loss functions and metrics for some supported compositional architectures
(See lego_blocks.py)
"""

import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import sklearn.metrics as skm
import matplotlib.pyplot as plt
import itertools

tfk=tf.keras
tfm=tf.math
tfkm=tfk.metrics
K=tfk.backend

eps=1E-6


########################################################################################################
'''
            Utilities and fast approximations        
''' 
########################################################################################################
def gammaln(x):
    # fast approximate gammaln from paul mineiro
    # http://www.machinedlearnings.com/2011/06/faster-lda.html
    logterm = tfm.log (x * (1.0 + x) * (2.0 + x))
    xp3 = 3.0 + x
    return -2.081061466 - x + 0.0833333 / xp3 - logterm + (2.5 + x) * tfm.log (xp3)

def logsumexp(self, vec1, vec2):
    flag = tf.greater(vec1, vec2)
    maxv = tf.where(flag, vec1, vec2)
    lse = tf.log(tf.exp(vec1 - maxv) + tf.exp(vec2 - maxv)) + maxv
    return lse

def inv_softplus_np(self, x):
    y = np.log(np.exp(x) - 1)
    return y 

########################################################################################################
'''
            Cross-entropy derivatives and advanced loss functions         
''' 
########################################################################################################

def stable_focal_loss(gamma=2, alpha=np.ones((3,1))):  
    def focal_loss(y_true, y_pred):#with tensorflow
        ### This part does the cross-entropy calculation without weighting
        #alpha=tf.constant(alpha,tf.float32)
        y_true=tf.cast(y_true,tf.float32)
        loss=tf.nn.sigmoid_cross_entropy_with_logits(y_true,y_pred)
       
        ### Now we compute the log of the uncertainty part
        ## y_pred is logit
        invprobs = tf.math.log_sigmoid(-y_pred * (y_true * 2 - 1))
       
        ### Then, we exponentiate it
        expinvprobs=tf.math.exp(invprobs*gamma)
       
        ### Then, we combine everything to form the focal loss
        floss=expinvprobs*loss
        return tf.math.reduce_mean(floss*alpha) ####Weighted mean of losses per taxa
    return focal_loss 



def bce(bw=None,lw=None,avg=False):
    print('Binary cross entropy with class/label weights')
    def bce_weighted(y_true,y_pred):
      # Compute cross entropy from probabilities.
      y_true=tf.cast(y_true,tf.float32)
      bcel = bw * y_true * tfm.log(y_pred + eps)
      bcel += (1 - y_true) * tfm.log(1 - y_pred + eps)
      return tfm.reduce_mean(-bcel * lw) if avg else -bcel*lw
  
    return bce_weighted


def negbin_loss(y_true,y_pred):   ###uses y_pred as log probability or logit or applies softplus=relu to it beforehand
    r=y_pred[1]
    p=y_pred[0]
    y_true=tf.cast(y_true,tf.float32)
    logprob = gammaln(y_true + r) - gammaln(y_true + 1.0) -  \
                 gammaln(r) + r * tfm.log(r) + \
                 y_true * tfm.log(p+eps) - (r + y_true) * tfm.log(r + p)

    return tfm.reduce_mean(logprob)

#### Loss functions selector ####
loss_fn={'normal':tfk.losses.mean_squared_error, ##assumes that outputs are scaled (sd=1)
        'poisson2':tfk.losses.poisson,
        'poisson':tfk.losses.poisson,
        'binomial':lambda w: bce(bw=w[0],lw=w[1],avg=True) if w is None else tfk.losses.binary_crossentropy,
        'binomial2':tfk.losses.binary_crossentropy,
        'categorical':tfk.losses.categorical_crossentropy,
        'negbin':negbin_loss,
        'negbin2':negbin_loss
        }

########################################################################################################
'''
                    Custom metrics for exponential family outputs        
''' 
########################################################################################################

def poly(x, p):
    x = np.array(x)
    X = np.transpose(np.vstack((x**k for k in range(p+1))))
    return np.linalg.qr(X)[0][:,1:]


def poisson_dev(y_true, y_pred):
    ### Adapted from Ron Richman's https://github.com/RonRichman/AI_in_Actuarial_Science/blob/master/poisson_dev.py
    K=tfk.backend
    y_true=K.cast(y_true,dtype=tf.float32)
    y_pred=K.cast(y_pred,dtype=tf.float32)
    return 2*K.mean(y_pred - y_true -y_true*(K.log(K.clip(y_pred,K.epsilon(),None)) -K.log(K.clip(y_true,K.epsilon(),None))),axis=-1)  


## Metric functions selector ##
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

########################################################################################################
'''
            Classification metrics for multi-label problems
                        Operates on numpy objects            
''' 
########################################################################################################

    
def eval_task(y_true,y_pred,taxa=None,th=0.5,prevs=None):  ### provide inputs as np array
    ### Returns a long dataframe with pairs (dataset, taxa, metric)
    if taxa is None:
        taxa=np.arange(y_true.shape[1])
        
    if prevs is None:
        prevs=np.ones((y_true.shape[1],1))
    
    results=[]
    for j in taxa:
        y_pred_class=(y_pred[:,j]>th).astype(int)
        y_test=y_true[:,j]
        
        acc=skm.accuracy_score(y_test,y_pred_class)
        conf=skm.confusion_matrix(y_test,y_pred_class,labels=[1,0])
        
        tp=conf[0,0]
        fn=conf[0,1]
        fp=conf[1,0]
        tn=conf[1,1]
        
        #acc=(tp+tn)/(tp+tn+fp+fn)
        
        if (tn+fp)==0:
            spec=1
        else:
            spec=tn/(tn+fp)

        if (tp+fn)==0:
            sens=rec=1
        
        else:
            sens=rec=tp/(tp+fn)
        
        if (tp+fp)==0:
            prec=1
        else:
            prec=tp/(tp+fp)
        
        if (prec+rec)==0:
            f1=0
            
        else:
            f1=2*prec*rec/(prec+rec)
        
        try:
            auc=skm.roc_auc_score(y_test,y_pred[:,j],labels=[1,0])
        except ValueError:
            auc=-1
        
        results.append({
            'taxa':j,
            'prev':prevs[j],
            'th':th,
            'accuracy':acc,
            'recall':rec,
            'precision':prec,
            'f1':f1,
            'sensitivity':sens,
            'specificity':spec,
            'tss':rec+spec-1,
            'roc_auc':auc,
            'conf':[tp,fn,fp,tn],
            })
        
    return results

def roc_multilabel(y_test,y_score,n_classes,file):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = skm.roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = skm.auc(fpr[i], tpr[i])
    
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = skm.roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = skm.auc(fpr["micro"], tpr["micro"])
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    
    # Finally average it and compute AUC
    mean_tpr /= n_classes
    
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = skm.auc(fpr["macro"], tpr["macro"])
    
    # Plot all ROC curves
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)
    
    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)
    
    colors = itertools.cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(i, roc_auc[i]))
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.savefig(file) 
    


def tss_score(cm):
    tn, fp, fn, tp = cm.ravel()
    spec=tn/(tn+fp)
    sens=tp/(tp+fn)
    
    return spec+sens-1


def ml_tss_score(y_true,y_pred,average=None):
    ml_cm = skm.multilabel_confusion_matrix(y_true,y_pred,labels=[0,1])
    if average=="micro":
        cm=ml_cm.sum(axis=0)
        return tss_score(cm)
    
    if average=="macro":
        tss_cum=0
        for j in range(ml_cm.shape[0]):
            cm_j=ml_cm[j,:,:]
            tss_cum+=tss_score(cm_j)
            
        return tss_cum/ml_cm.shape[0]
    
    if average=="weighted":
        weights=y_true.sum(axis=0)/y_true.shape[0]
        tss_cum=0
        for j in range(ml_cm.shape[0]):
            cm_j=ml_cm[j,:,:]
            tss_cum+=tss_score(cm_j) 
            
        return (tss_cum*weights)/np.sum(weights)
    
    else:
        tss_score(skm.confusion_matrix(y_true,y_pred))
        
            
    
    
    
metric_meta={ ## met_name:(use_th,has_average_mode,minOrMax,notonlyMLL,function_to_call)
    'balanced_accuracy':(True,False,'max',True,skm.balanced_accuracy_score), 
    'auc':(False,True,'max',True,skm.roc_auc_score),
    'recall':(True,True,'max',True,skm.recall_score),
    'precision':(True,True,'max',True,skm.precision_score),
    'f1':(True,True,'max',True,skm.f1_score),
    'tss':(True,True,'max',True,ml_tss_score),
    'coverage':(False,False,'min',False,skm.coverage_error),
    }
    
def score_j(mhsm,pred_fun,j=None,metric="balanced_accuracy",mode=None,m=1,th=0.5):
    meta=metric_meta.get(metric)
    
    metric_func=meta[4]
    
    if type(j)==list:
        ltaxa=j
        
    elif type(j)==int:
        ltaxa=[j]
        
    else:
        ## All taxa
        ltaxa=np.arange(m)
        
    def score(X, y):    ### Single task
        print("Single task score")
        y_pred=pred_fun(X) 
        
        if meta[0]:
            y_pred_class=(y_pred>th).astype(int)
            
        else:
            y_pred_class=y_pred
        
        score_sl=metric_func(y[:,j],y_pred_class[:,j])

        return score_sl   

    def score_mll(X, y):
        #print("Average "+mode+" "+metric +" score")
        y_pred=pred_fun(X)
        if meta[0]:
            y_pred_class=(y_pred>th).astype(int)
        else:
            y_pred_class=y_pred
        
        if mode in ['micro','macro','weighted']:
            score_ml=metric_func(y[:,ltaxa],y_pred_class[:,ltaxa],average=mode)
            
        else:
            print("Multiple task-wise scores")
            score_ml=np.array([metric_func(y[:,j],y_pred_class[:,j]) for j in ltaxa])

        return score_ml
    
    
    if len(ltaxa)==1:
        return score

    else:
        return score_mll 