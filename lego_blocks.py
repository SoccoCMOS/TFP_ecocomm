# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 10:22:15 2020

@author: saras

This script contains:
    A set of useful parameterizable components to plug and play in your neural architecture.
    
Each function has the following parameters:
    - Layers names - needs to be specified
    - Layers dimensions - needs to be specified
    - Regularization - can use default (no regularization)
    - Initialization - can use default (uniform glorot)
    
"""

"""
Dependencies: tensorflow 2.1, Keras (tf.keras), Tensorflow probability
"""
import tensorflow as tf
import tensorflow_probability as tfp


tfk = tf.keras
tfkl = tf.keras.layers
tfpl = tfp.layers
tfd = tfp.distributions
tfkv= tfk.utils
tfkr=tfk.regularizers
tfki=tfk.initializers

"""
Utility functions
"""
def get_regularizer(regtype,regparams):
    if(regtype=="l1"):
        kr=tfkr.l1(l=regparams[0])
    elif(regtype=="l2"):
        kr=tfkr.l2(l=regparams[0])
    elif(regtype=="l1_l2"):
        kr=tfkr.l1_l2(l1=regparams[0],l2=regparams[1])
    else:
        kr=None 
    return kr


def probit(x):
    normal = tfd.Normal(loc=0.,
                    scale=1.)
    return normal.cdf(x)

"""
Basic building blocks
- Heterogeneous inputs
- Fully connected architecture
- Heterogeneous outputs
"""

'''
Heterogeneous inputs
'''

def custom_input(in_config,concat_feat=True,model_name="model"):
    '''
    Collect inputs, embed categorical inputs and concatenate all to get features
    
    '''
    l_in=[]
    l_feat=[]
    
    for v in in_config:
        k=v[0]
        if k=='num':
            in_num=tfk.Input(shape=(v[1].get('dim'),),dtype=tf.float32,name=v[1].get('id'))
            l_in.append(in_num)
            l_feat.append(in_num)
        
        if k=='cat':
            for cv in v[1]:
                in_cat=tfk.Input(shape=(1,),dtype=tf.int32,name=cv.get('id'))
                emb_cat=tfkl.Flatten()(
                    tfkl.Embedding(cv.get('mod'),cv.get('emb'),name=cv.get('id')+"_emb")(in_cat))
                
                l_in.append(in_cat)
                l_feat.append(emb_cat)
                
    
    if (len(l_feat)>1):
        if concat_feat:
            in_feat=tfkl.Concatenate(name=model_name+"_featVector")(l_feat)
        else:
            in_feat=l_feat
        
    elif len(l_feat)==1:
        in_feat=l_feat[0]
        
    else:
        raise BaseException('ModelSpecificationError: check input data configuration')
        
    
    return l_in, in_feat 


'''
Fully connected architecture with single input, single output
'''
def fc_nn(name,archi,reg=None):
    '''
    Syntax
    ********
    name="model_name"
    archi={'nbnum':1,'nl':2,'nn':[10,23]}
    reg={'regtype':'l1','regparam':(0.01,0.01),'dropout':[1,0.8]}
    '''  
    
    in_num=tfk.Input(shape=(archi.get('nbnum'),),name=name+"_input",dtype=tf.float32)
    if reg is None:
        reg={'regtype':None,'regparam':None}
        
    kreg=get_regularizer(reg.get("regtype"),reg.get("regparam"))
    
    prev=in_num
    
    activs=archi.get("activation")
    if type(activs)!=list:
        activs=[activs]*archi.get("nl") ##same activation everywhere
        raise Warning('Using the same activation for all layers.')
    
    ### Whether to dropout inputs: useful in case of images ###    
    if reg.get('dropout') is not None:
            rate=reg.get("dropout")[0]
            if rate<1:
                prev=tfkl.Dropout(rate)(prev)
    
    for i in range(archi.get("nl")):
        prev=tfkl.Dense(archi.get("nn")[i], activation=activs[i],name=name+"_"+str(i),kernel_regularizer=kreg)(prev)
        if reg.get('dropout') is not None:
            rate=reg.get("dropout")[i+1]
            if rate<1:
                prev=tfkl.Dropout(rate)(prev)
    
    #out=tfkl.Activation(activation=archi.get("o_activation") if archi.get("activation") is not None else None)
    m=tfk.Model(in_num,prev,name=name)
    
    return(m)
 

'''
Heterogeneous outputs
'''
def mtl_output(shared_config,out_configs,model_name="mtl"):
    
    ### Shared feature transformation component
    shared=fc_nn(name=shared_config['name'],archi=shared_config['archi'],reg=shared_config['reg'])
    l_outputs=[]

    
    ### Task-specific components
    for tc in out_configs:  ##out_config is a list of configs for each output {'type':'cat','specific':{'archi','reg'}}
        tnm=tc['name']
        specific_config=tc['specific']
        tc_fc=fc_nn(tnm,specific_config['archi'],reg=specific_config['reg'])
        
        act=tc['activation']
        if act is None: ##specify default value
            if tc['type']=='binary':
                act='sigmoid'
                act_=act
                
            elif tc['type']=='categorical':
                act='softmax'
                act_=act
                
            else:
                act='linear'
                act_=act
        
        if act=='probit':
            act=probit
            act_='probit'
            
        t_pred=tfkl.Activation(activation=act,name=tnm+"_"+act_)(tc_fc(shared.output))
        
        l_outputs.append(t_pred)
    
    return tfk.Model(shared.inputs,l_outputs,name=model_name)
   

"""
Compositional architectures

1. Autoencoders (generic case with multiple types of inputs/outputs)
2. Variational autoencoders
3. Conditional variational autoencoders
4. Multi-task hard sharing (without the encoder part)
5. Multi-task soft sharing (without the encoder part but soft sharing)
"""

def autoencoder(in_config,encoder_config,decoder_config,out_config):
    l_in,l_feat=custom_input(in_config,concat_feat=True,model_name='in_autoenc')
    encoder_net=fc_nn(name='encoder',archi=encoder_config['archi'],reg=encoder_config['reg'])
    decoder_net=mtl_output(decoder_config,out_config,model_name='decoder')

    autoencoder=tfk.Model(l_in,decoder_net(encoder_net(l_feat)))
    
    return encoder_net, decoder_net, autoencoder



def code_generator(encoded_size,prior,latent_params,beta=1.0):  
    
    out_code=tfpl.MultivariateNormalTriL(
        encoded_size,
        activity_regularizer=tfpl.KLDivergenceRegularizer(prior,weight=beta))(latent_params)
    return(out_code)



def variational_autoencoder(in_config,encoder_config,decoder_config,out_config,prior=None,beta=1.0):
    ### Inputs
    l_in,l_feat=custom_input(in_config,concat_feat=True,model_name='in_vae')
    
    ### Getting the dimensions right
    encoded_size=encoder_config['archi'].get('nn')[-1]
    paramsize=tfpl.MultivariateNormalTriL.params_size(encoded_size)
    
    ### Updating the encoder output's parameter 
    encoder_config['archi'].get('nn')[-1]=paramsize
    
    ### Recognition
    encoder_net=fc_nn(name='encoder',archi=encoder_config['archi'],reg=encoder_config['reg'])
    latent_params=encoder_net(l_feat)
    
    if prior is None:
        ### Prior
        prior = tfd.Independent(tfd.Normal(loc=tf.zeros(encoded_size), scale=1),
                        reinterpreted_batch_ndims=1)
    
    ### Generate code
    latent_code=code_generator(encoded_size,prior,latent_params,beta)
    
    ### Decoder network
    decoder_net=mtl_output(decoder_config,out_config,model_name='decoder')
    l_out=decoder_net(latent_code)
    
    var_autoencoder=tfk.Model(l_in,l_out)
    
    return encoder_net, decoder_net, prior, var_autoencoder


def conditional_variational_autoencoder(in_feat,in_config,fe_config,encoder_config,decoder_config,out_config):
    '''
    Prior network
    '''
    
    ###Input features ###
    l_in_E,l_E=custom_input(in_feat,concat_feat=True,model_name='in_fe')
    
    ###Setting up the dimensions
    
    encoded_size=encoder_config['archi'].get('nn')[-1]
    paramsize=tfpl.MultivariateNormalTriL.params_size(encoded_size)
    
    ### Updating the encoder output's parameter 
    fe_config['archi'].get('nn')[-1]=paramsize
    
    ###Feature extraction network
    fe_net=fc_nn(name='fe',archi=fe_config['archi'],reg=fe_config['reg'])
    X=fe_net(l_E)
    
    prior=tfpl.MultivariateNormalTriL(encoded_size)(X)
    
    '''
    VAE
    '''
    ### Inputs
    l_in,l_feat=custom_input(in_config,concat_feat=True,model_name='in_vae')
    
    
    ### Updating the encoder output's parameter 
    encoder_config['archi'].get('nn')[-1]=paramsize
    
    ### Recognition
    encoder_net=fc_nn(name='encoder',archi=encoder_config['archi'],reg=encoder_config['reg'])
    latent_params=encoder_net(l_feat)
    
    ### Generate code with regularization
    latent_code=code_generator(encoded_size,prior,latent_params)
    
    ### Decoder network
    decoder_net=mtl_output(decoder_config,out_config,model_name='decoder')
    
    in_decod=tfkl.Concatenate()([l_E,latent_code])
    l_out=decoder_net(in_decod)
    
    cvar_autoencoder=tfk.Model(l_in+l_in_E,l_out)
    
    return encoder_net, decoder_net, tfk.Model(l_in_E,prior), cvar_autoencoder
    
    
    

"""
Unit test

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
Unit test

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
    
    
"""
Unit test
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


u_custom_inputs()
"""     


"""
Unit test
def u_fc_nn():
    fc_nn('model_name',{'nbnum':5,'nl':0,'nn':[]}).summary()
    fc_nn('model_name',{'nbnum':5,'nl':2,'nn':[10,10]}).summary()
    fc_nn('model_name',{'nbnum':6,'nl':5,'nn':[6]*10},{'regtype':'l1','regparam':(0.01,0.01),'dropout':[1,0.8,0.8,0.8,0.8,1]}).summary()
    fc_nn('model_name',{'nbnum':3,'nl':2,'nn':[6]*2},{'regtype':None,'regparam':None,'dropout':[1,1,0.8]}).summary()


u_fc_nn() 

"""  


"""
Unit tests

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


u_mtl_output()

""" 

"""
Unit test

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
        
"""        
    
    
    
    



