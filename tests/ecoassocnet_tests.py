# -*- coding: utf-8 -*-
"""
Created on Sun May 10 15:47:16 2020

@author: saras
"""


from ecoassocnet import *   
from model_selection import *
        
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

def ut_ecoassocnet():        
    d=6
    m=1000
    p=5
    c=0
    n=64
    
    ### Generate data to test fitting ###
    x=np.random.normal(0,1,(n,p))
    y_true=np.random.poisson(1,(n,m))
    
    shared_config={'latent_dim':d,'pool_size':m,'reg':(0.,0.0),'dist':('negbin',None)}
    hsm_config={'input':[('num',{'id':'num1','dim':p})],'reg':None,'archi':{'nbnum':p,'nl':0,'nn':[],'activation':'relu','fit_bias':False}}
    im_config={'input':[],'sym':True,'reg':None,'reg':None,
                'archi':{'nbnum':c,'nl':1,'nn':[d],'activation':'relu','fit_bias':False}}
    
    
    train_config={'bsize':512,'max_iter':200,
                  'tol':1E-3,'patience':5,
                  'mode':'min','objective':'loss'}
    
    
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

#########################################################################################################################
'''
Model selection
'''
########################################################################################################################

def debug_assoc():
    m=4
    d=2
    alpha=np.random.uniform(-1,1,(m,d))
    alpha_n=alpha/np.linalg.norm(alpha,axis=1)[:,np.newaxis]
    
    assoc=np.dot(alpha_n,alpha_n.T)
    
    t_alpha=tf.constant(alpha_n)
    
    tfk.backend.dot(t_alpha,tf.transpose(t_alpha))
    #tf.tensordot(t_alpha,tf.transpose(t_alpha))


import matplotlib.pyplot as plt    

def ut_stars():
    m=50
    p=2
    c=0
    n=1000
    x=np.random.normal(0,1,(n,p))
    beta=np.random.uniform(-1,1,(p,m))
    y=(np.dot(x,beta)>0).astype(int)
    
    data=(x,y,None)
    ea_msel=EAModelSelection(data,'binomial2',True,False)   
    ea_msel.generate_grid(d_range=[25],l_range_hsm=[0.],l_range_ass=[0.,0.0001,0.001,0.01,0.1,1],l_range_im=[0.],max_it=200)
    
    star_unorm=ea_msel.stars_network(0.05,0.0001,0.01,N=2,reg_emb=False,norm_emb=False)
    star_norm=ea_msel.stars_network(0.05,0.0001,0.01,N=20,reg_emb=False,norm_emb=True)

def ut_reg():
    m=50
    p=2
    c=0
    n=1000
    x=np.random.normal(0,1,(n,p))
    beta=np.random.uniform(-1,1,(p,m))
    y=(np.dot(x,beta)>0).astype(int)
    
    data=(x,y,None)
    ea_msel=EAModelSelection(data,'binomial2',True,False)   
    ea_msel.generate_grid(d_range=[int(m/2)],l_range_hsm=[0.],l_range_ass=[0.,0.001,0.01,0.1,1],l_range_im=[0.],max_it=200)
    
    print('With embedding regularization and unit norm constraints')
    path_emb_norm=ea_msel.lasso_path_network(N=2,reg_emb=True,norm_emb=True)
    data=np.array([(x['lambda'],compute_density(x['associations']),compute_effective_dimension(x['embeddings']['response'])) for x in path_emb_norm])
    plt.plot(data[:,0],data[:,1])
    plt.title('Density Embedding, regularized, normed embeddings')
    
    plt.plot(data[:,0],data[:,2])
    plt.title('Dimension Embedding, regularized, normed embeddings')
    
    
    print('With embedding regularization without unit norm constraints')
    path_emb_unorm=ea_msel.lasso_path_network(N=2,reg_emb=True,norm_emb=False)
    data=np.array([(x['lambda'],compute_density(x['associations']),compute_effective_dimension(x['embeddings']['response'])) for x in path_emb_unorm])
    plt.plot(data[:,0],data[:,1])
    plt.title('Density Embedding, regularized, unormed embeddings')
    plt.close()
    plt.plot(data[:,0],data[:,2])
    plt.title('Dimension Embedding, regularized, unormed embeddings')
    plt.close()
    
    print('With association regularization and unit norm constraints')
    path_assoc_norm=ea_msel.lasso_path_network(N=2,reg_emb=False,norm_emb=True)
    data=np.array([(x['lambda'],compute_density(x['associations']),compute_effective_dimension(x['embeddings']['response'])) for x in path_assoc_norm])
    plt.plot(data[:,0],data[:,1])
    plt.title('Density associations, regularized, normed embeddings')
    plt.close()
    plt.plot(data[:,0],data[:,2])
    plt.title('Dimension associations, regularized, normed embeddings')
    plt.close()
    
    print('With association regularization without unit norm constraints')
    path_assoc_unorm=ea_msel.lasso_path_network(N=2,reg_emb=False,norm_emb=False)
    data=np.array([(x['lambda'],compute_density(x['associations']),compute_effective_dimension(x['embeddings']['response'])) for x in path_assoc_unorm])
    plt.plot(data[:,0],data[:,1])
    plt.title('Density Associations regularized, unnormed embeddings')
    plt.close()
    plt.plot(data[:,0],data[:,2])
    plt.title('Dimension Associations regularized, unnormed embeddings')
    plt.close()
    #grid=ea_msel.param_grid
    
    #cpt=0
    #conf={k:grid.get(k)[cpt] for k in grid.keys()}    
    #ea_stars=StARS(conf)      