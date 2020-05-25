# -*- coding: utf-8 -*-
"""
Created on Sat May  2 11:38:16 2020

@author: saras
"""

from lego_blocks import *

"""
Unit test Conditional Variational Autoencoder

"""

def u_cvae():
    in_feat=[('num',{'id':'env','dim':13})]
    in_config=[('num',{'id':'taxa','dim':86})]
    
    fe_config={'name':'fe','archi':{'nbnum':13,'nl':1,'nn':[5],'activation':'relu'},'reg':None}
    
    
    encoder_config={'name':'encoder','archi':{'nbnum':86,'nl':1,'nn':[16],'activation':'relu'},'reg':None}
    decoder_config={'name':'shared_decoder','archi':{'nbnum':16+13,'nl':0,'nn':[],'activation':'relu'},'reg':None}
    
    
    taxa={'name':'specific_decoder','archi':{'nbnum':16+13,'nl':1,'nn':[86],'activation':'relu'},'reg':None}
    
    out_config=[ {'name':'t','type':'binary','specific':taxa,'activation':'probit'}]    
    
    e, d, p, cvae=conditional_variational_autoencoder(in_feat,in_config,fe_config,encoder_config,decoder_config,out_config)
    
    tfkv.plot_model(e)
    tfkv.plot_model(d)
    tfkv.plot_model(p)
    tfkv.plot_model(cvae)


"""
Unit test Variational autoencoder

"""

def u_vae():
    in_config=[('num',{'id':'taxa','dim':86})]
    
    encoder_config={'name':'encoder','archi':{'nbnum':86,'nl':1,'nn':[16],'activation':'relu'},'reg':None}
    decoder_config={'name':'shared_decoder','archi':{'nbnum':16,'nl':0,'nn':[],'activation':'relu'},'reg':None}
    
    
    taxa={'name':'specific_decoder','archi':{'nbnum':16,'nl':1,'nn':[86],'activation':'relu'},'reg':None}
    
    out_configs=[ {'name':'t','type':'binary','specific':taxa,'activation':'probit'}]
    
    
    e, d, p, vae=variational_autoencoder(in_config,encoder_config,decoder_config,out_configs)
    
    tfkv.plot_model(e,show_layer_names=True,show_shapes=True)
    tfkv.plot_model(d,show_layer_names=True,show_shapes=True)
    tfkv.plot_model(vae,show_layer_names=True,show_shapes=True)
    
    
    '''
    Heterogeneous  tasks
    '''
    in_config=[('num',{'id':'num1','dim':3}),
               ('cat',[{'id':'cat1','mod':5,'emb':2},{'id':'cat2','mod':7,'emb':4}])]
    
    encoder_config={'name':'encoder','archi':{'nbnum':9,'nl':1,'nn':[3],'activation':'relu'},'reg':None}
    decoder_config={'name':'shared_decoder','archi':{'nbnum':3,'nl':0,'nn':[],'activation':'relu'},'reg':None}
    
    
    sp1={'name':'specific_decoder','archi':{'nbnum':3,'nl':1,'nn':[2],'activation':'relu'},'reg':None}
    sp2={'name':'specific_decoder','archi':{'nbnum':3,'nl':1,'nn':[1],'activation':'relu'},'reg':None}
    sp3={'name':'specific_decoder','archi':{'nbnum':3,'nl':1,'nn':[5],'activation':'relu'},'reg':None}
    sp4={'name':'specific_decoder','archi':{'nbnum':3,'nl':1,'nn':[7],'activation':'relu'},'reg':None}
    
    out_config=[{'name':'t1','type':'numeric','specific':sp1,'activation':None},
                {'name':'t2','type':'binary','specific':sp2,'activation':'probit'},
                 {'name':'t3','type':'categorical','specific':sp3,'activation':None},
                 {'name':'t4','type':'categorical','specific':sp4,'activation':None}]
    
    
    e, d, p, vae=variational_autoencoder(in_config,encoder_config,decoder_config,out_config)
    
    tfkv.plot_model(e,show_layer_names=True,show_shapes=True)
    tfkv.plot_model(d,show_layer_names=True,show_shapes=True)
    tfkv.plot_model(vae,show_layer_names=True,show_shapes=True)
        

"""      
 Unit test Custom inputs   
"""

def u_custom_inputs():
    in_config1=[]
    in_config2=[('num',{'id':'num','dim':3,'act':'relu'})]
    in_config3=[('cat',[{'id':'cat1','mod':5,'emb':2},{'id':'cat2','mod':7,'emb':4}])]
    in_config4=in_config2+in_config3
    
    for inc in [in_config2,in_config3,in_config4]:
        print(inc)
        custom_input(inc,False)
        custom_input(inc,True)
        
    custom_input(in_config1)


"""     
Unit test Fully-connected NN
"""

def u_fc_nn():
    fc_nn('model_name',{'nbnum':5,'nl':0,'nn':[]}).summary()
    fc_nn('model_name',{'nbnum':5,'nl':2,'nn':[10,10]}).summary()
    fc_nn('model_name',{'nbnum':6,'nl':5,'nn':[6]*10},{'regtype':'l1','regparam':(0.01,0.01),'dropout':[1,0.8,0.8,0.8,0.8,1]}).summary()
    fc_nn('model_name',{'nbnum':3,'nl':2,'nn':[6]*2},{'regtype':None,'regparam':None,'dropout':[1,1,0.8]}).summary()

"""  
Unit tests MTL output
"""

def u_mtl_output():
    shared_config={'name':'sh_layer','archi':{'nbnum':6,'nl':1,'nn':[10],'activation':'relu'},'reg':None}
    sp1={'name':'sh_layer','archi':{'nbnum':10,'nl':1,'nn':[2],'activation':'relu'},'reg':None}
    sp2={'name':'sh_layer','archi':{'nbnum':10,'nl':1,'nn':[3],'activation':'relu'},'reg':None}
    sp3={'name':'sh_layer','archi':{'nbnum':10,'nl':1,'nn':[4],'activation':'relu'},'reg':None}
    out_configs=[{'name':'t1','type':'numeric','specific':sp1,'activation':None},
                 {'name':'t2','type':'categorical','specific':sp2,'activation':None},
                 {'name':'t3','type':'binary','specific':sp3,'activation':None}
                 ]
    
    model=mtl_output(shared_config,out_configs)
    model.summary()
    
    tfkv.plot_model(model)

""" 
Unit test  autoencoder
"""

def u_autoencoder():
    '''
    Heterogeneous  tasks
    '''
    in_config=[('num',{'id':'num1','dim':3}),
               ('cat',[{'id':'cat1','mod':5,'emb':2},{'id':'cat2','mod':7,'emb':4}])]
    
    encoder_config={'name':'encoder','archi':{'nbnum':9,'nl':1,'nn':[3],'activation':'relu'},'reg':None}
    decoder_config={'name':'shared_decoder','archi':{'nbnum':3,'nl':0,'nn':[],'activation':'relu'},'reg':None}
    
    
    sp1={'name':'specific_decoder','archi':{'nbnum':3,'nl':1,'nn':[2],'activation':'relu'},'reg':None}
    sp2={'name':'specific_decoder','archi':{'nbnum':3,'nl':1,'nn':[1],'activation':'relu'},'reg':None}
    sp3={'name':'specific_decoder','archi':{'nbnum':3,'nl':1,'nn':[5],'activation':'relu'},'reg':None}
    sp4={'name':'specific_decoder','archi':{'nbnum':3,'nl':1,'nn':[7],'activation':'relu'},'reg':None}
    
    out_config=[{'name':'t1','type':'numeric','specific':sp1,'activation':None},
                {'name':'t2','type':'binary','specific':sp2,'activation':None},
                 {'name':'t3','type':'categorical','specific':sp3,'activation':None},
                 {'name':'t4','type':'categorical','specific':sp4,'activation':None}]
    
    
    e, d, ae=autoencoder(in_config,encoder_config,decoder_config,out_config)
    
    tfkv.plot_model(e,show_layer_names=True,show_shapes=True)
    tfkv.plot_model(d,show_layer_names=True,show_shapes=True)
    tfkv.plot_model(ae,show_layer_names=True,show_shapes=True)
    
    
    '''
    Joint  bernoulli 
    '''
    in_config=[('num',{'id':'taxa','dim':86})]
    
    encoder_config={'name':'encoder','archi':{'nbnum':86,'nl':1,'nn':[16],'activation':'relu'},'reg':None}
    decoder_config={'name':'shared_decoder','archi':{'nbnum':16,'nl':0,'nn':[],'activation':'relu'},'reg':None}
    
    
    taxa={'name':'specific_decoder','archi':{'nbnum':16,'nl':1,'nn':[86],'activation':'relu'},'reg':None}
    
    out_config=[ {'name':'t','type':'binary','specific':taxa,'activation':'probit'}]
    
    
    e, d, ae=autoencoder(in_config,encoder_config,decoder_config,out_config)
    
    tfkv.plot_model(e,show_layer_names=True,show_shapes=True)
    tfkv.plot_model(d,show_layer_names=True,show_shapes=True)
    tfkv.plot_model(ae,show_layer_names=True,show_shapes=True)
    
    '''
    Independent  bernoulli 
    '''
    in_config=[('num',{'id':'taxa','dim':86})]
    
    encoder_config={'name':'encoder','archi':{'nbnum':86,'nl':1,'nn':[16],'activation':'relu'},'reg':None}
    decoder_config={'name':'shared_decoder','archi':{'nbnum':16,'nl':0,'nn':[],'activation':'relu'},'reg':None}
    
    
    taxa={'name':'specific_decoder','archi':{'nbnum':16,'nl':1,'nn':[1],'activation':'relu'},'reg':None}
    
    out_config=[ {'name':'taxa_%d'%i,'type':'binary','specific':taxa,'activation':None} for i in range(86)]
    
    
    e, d, ae=autoencoder(in_config,encoder_config,decoder_config,out_config)
    
    tfkv.plot_model(e,show_layer_names=True,show_shapes=True)
    tfkv.plot_model(d,show_layer_names=True,show_shapes=True)
    tfkv.plot_model(ae,show_layer_names=True,show_shapes=True)
        


u_custom_inputs()
u_fc_nn() 
u_vae()
u_cvae()
u_mtl_output()
u_autoencoder()        
    
    
    
    



