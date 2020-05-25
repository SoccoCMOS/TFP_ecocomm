# -*- coding: utf-8 -*-
"""
Created on Sun May 10 15:47:16 2020

@author: saras
"""


from ecoassocnet import *           
#########################################################################################################################
'''
                            Hyperparameter tuning for EcoAssocNet
                            
 Hyperparamaters that could/should be tuned  :
     1) latent dimension d : integer [2,m/2]
     2) symmetric response/effect : boolean (True,False)
     3) regularization :
        - L1 weight  (set to 0 if no l1 is done)
        - L2 weight (set to 0 is no l2 is done)
     4) Whether to mask diagonals in computing biotic context embedding: boolean ([True], False)
     5) Distribution among those applicable to your data + hyperparameter
     6) HSM architecture hyperparams
     7) Output activation among those applicable
     8) Loss and metrics
    
    Training stuffs
    8) Optimizer
    9) Batch_size
    
                          
        '''
#########################################################################################################################        
## Setting up an HParams experiment for a given dataset
        

#########################################################################################################################
''' 
                                        Unit tests 
        '''
#########################################################################################################################
d=6
m=1000
p=5
c=0
n=64

shared_config={'latent_dim':d,'pool_size':m,'reg':(0.,0.0),'dist':('negbin',None)}
hsm_config={'input':[('num',{'id':'num1','dim':p})],'reg':None,'archi':{'nbnum':p,'nl':0,'nn':[],'activation':'relu','fit_bias':False}}
im_config={'input':[],'sym':True,'reg':None,'reg':None,
            'archi':{'nbnum':c,'nl':1,'nn':[d],'activation':'relu','fit_bias':False}}


train_config={'bsize':512,'max_iter':200,
              'tol':1E-3,'patience':5,
              'mode':'min','objective':'loss'}


### Generate data to test fitting ###
x=np.random.normal(0,1,(n,p))
y_true=np.random.poisson(1,(n,m))

ea=EcoAssocNet(model_name="randomtest",shared_config=shared_config,hsm_config=hsm_config,im_config=im_config)
ea.create_architecture() 
tfkv.plot_model(ea.pred_model,show_shapes=True)
ea.compile_model()  
y_pred=ea.pred_model.predict([x,y_true])

print('Initial loss %.3f' % ea.nbr.loss(y_true, y_pred))

w_init=ea.pred_model.get_weights()
r_init=ea.nbr.theta_variable.numpy()

cbk=[tfk.callbacks.TensorBoard(write_graph=False,
                                #embeddings_layer_names=['association_embedding/taxa_embedding/latent_0'],
                                update_freq='epoch',
                                write_images=False,
                                histogram_freq=5)]
                                
                                
import time
start_time = time.time()
ea.fit_model({'x':x,'y':y_true},None,train_config,2,[])
print("--- %s seconds ---" % (time.time() - start_time))

## Checking updates of dispersion parameter
w_last=ea.pred_model.get_weights()
r_last=ea.nbr.theta_variable.numpy()

# w=ea.pred_model.get_weights()
# wa=ea.association.get_weights()

# mat=ea.association.assoc.numpy()

#########################################################################################################################
