import os
import sys
import numpy as np
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import LeakyReLU, Conv1D, Input, Flatten, Conv2DTranspose, Lambda, Conv2D, Reshape
from keras.layers import Dense, Activation, Dropout, Reshape, Permute
from keras.layers import GRU, LSTM
from keras.optimizers import Adam
from keras import backend as K
from discrimination import MinibatchDiscrimination
from keras.constraints import Constraint

def sigmoid_crossentropy(y_true, y_pred):
    loss = K.mean(K.maximum(y_pred,0) - y_true*y_pred + K.log(1+K.exp(-K.abs(y_pred))))
    return loss

def wassertein_loss(y_true,y_pred):
    loss = -K.mean(y_pred*y_true)+K.mean((1-y_true)*y_pred)
    return loss

class WeightClipping(Constraint):
    def __init__(self,clip_value):
        self.clip_value = clip_value

    def __call__(self,weights):
        return K.clip(weights,-self.clip_value,self.clip_value)
    
    def get_config(self):
        return{'clip_value':self.clip_value}

def build_discriminator(data_params,network_params):
    ####input_signal = Input(shape=(data_params['ntime'],data_params['nfreq'],1))
    input_signal = Input(shape=(data_params['ntime'],data_params['nfreq'],2))
    if network_params['dcgan_loss'] == 'wgan':
        ClipConst = WeightClipping(0.1)
    else:
        ClipConst = None
    # 1st layer [128,128,2]->[64,64,64]
    fake = Conv2D(64,kernel_size=(5,5),strides=(2,2),padding='same',data_format='channels_last',kernel_constraint=ClipConst)(input_signal)
    fake = LeakyReLU(alpha=0.2)(fake)
    # 2nd layer [64,64,64]->[32,32,128]
    fake = Conv2D(128,kernel_size=(5,5),strides=(2,2),padding='same',data_format='channels_last',kernel_constraint=ClipConst)(fake)
    fake = LeakyReLU(alpha=0.2)(fake)
    # 3rd layer [32,32,128]->[16,16,256]
    fake = Conv2D(256,kernel_size=(5,5),strides=(2,2),padding='same',data_format='channels_last',kernel_constraint=ClipConst)(fake)
    fake = LeakyReLU(alpha=0.2)(fake)
    # 4th layer [16,16,256]->[8,8,512]
    fake = Conv2D(512,kernel_size=(5,5),strides=(2,2),padding='same',data_format='channels_last',kernel_constraint=ClipConst)(fake)
    fake = LeakyReLU(alpha=0.2)(fake)
    # 5th layer [8,8,512]->[4,4,1024]
    fake = Conv2D(1024,kernel_size=(5,5),strides=(2,2),padding='same',data_format='channels_last',kernel_constraint=ClipConst)(fake)
    fake = LeakyReLU(alpha=0.2)(fake)
    fake = Flatten()(fake)
    if network_params['use_MBdisc']:
        fake = MinibatchDiscrimination(network_params['MD_nb_kernels'],network_params['MD_kernel_dim'],input_dim=16*1024)(fake)
    fake = Dense(1,activation=None)(fake)
    Dis = Model(inputs=input_signal,outputs=fake)
    return(Dis)

def build_generator(data_params,network_params):
    input_noise = Input(shape=(network_params['latent_dim'],))
    # FC and reshape [100,]->[4,4,1024]
    fake_signal = Dense(256*64,activation=None,use_bias=False)(input_noise)
    fake_signal = Reshape((-1,4,1024))(fake_signal)
    fake_signal = Activation('relu')(fake_signal)
    # 1st layer [4,4,1024]->[8,8,512]
    fake_signal = Conv2DTranspose(512,kernel_size=(5,5),strides=(2,2),padding='same',data_format='channels_last',use_bias=False)(fake_signal)
    fake_signal = Activation('relu')(fake_signal)
    # 2nd layer [8,8,512]->[16,16,256]
    fake_signal = Conv2DTranspose(256,kernel_size=(5,5),strides=(2,2),padding='same',data_format='channels_last',use_bias=False)(fake_signal)
    fake_signal = Activation('relu')(fake_signal)
    # 3rd layer [16,16,256]->[32,32,128]
    fake_signal = Conv2DTranspose(128,kernel_size=(5,5),strides=(2,2),padding='same',data_format='channels_last',use_bias=False)(fake_signal)
    fake_signal = Activation('relu')(fake_signal)
    # 4th layer [32,32,128]->[64,64,64]
    fake_signal = Conv2DTranspose(64,kernel_size=(5,5),strides=(2,2),padding='same',data_format='channels_last',use_bias=False)(fake_signal)
    fake_signal = Activation('relu')(fake_signal)
    # 5th layer [64,64,64]->[128,128,2]
    #fake_signal = Conv2DTranspose(1,kernel_size=(5,5),strides=(2,2),padding='same',data_format='channels_last',use_bias=False)(fake_signal)
    fake_signal = Conv2DTranspose(2,kernel_size=(5,5),strides=(2,2),padding='same',data_format='channels_last',use_bias=False)(fake_signal)
    fake_signal = Activation('tanh')(fake_signal)
    Gen = Model(inputs=input_noise,outputs=fake_signal)
    return(Gen)


if __name__ == "__main__":
    data_params = {'data_len':256*32,'nb_channel':1,'data_size':7000,'ntime':128,'nfreq':128}
    network_params = {'hidden_unit':100,'latent_dim':100,'dropout_rate':0,'batch_size':5,'dcgan_loss':'wgan','use_batchnorm':False,
                        'use_MBdisc':True,'MD_nb_kernels':10,'MD_kernel_dim':40}
    Generator = build_generator(data_params,network_params)
    Discriminator = build_discriminator(data_params,network_params)

    print(Generator.summary())
    print(Discriminator.summary())
    noise = Input(shape=(network_params['latent_dim'],))
    generated = Generator(noise)
    fake = Discriminator(generated)
    Discriminator.trainable = False

    Combined = Model(inputs=noise,outputs=fake)
    print(Combined.layers[2])


#Single connected layer: https://stackoverflow.com/questions/56825036/make-a-non-fully-connected-singly-connected-neural-network-in-keras
        
