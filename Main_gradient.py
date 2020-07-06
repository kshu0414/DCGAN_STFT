import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import os.path
import math
import random
import tensorflow as tf
import numpy as np
from keras.models import Sequential
from DCGAN_model import build_generator, build_discriminator, sigmoid_crossentropy, wassertein_loss
from utils import sine_wave, draw_train_loss, draw_generated_signal, draw_D_loss, get_batch_signal
from STFT import STFT, ISTFT, get_STFT_moment, drawSTFTSpec, STFT_2C, ISTFT_2C
from metrics import pairwisedistances,pdist,MMD, RMSE
from keras.optimizers import Adam, SGD
from keras.models import Model, load_model
from keras.layers import Input
from keras.utils.generic_utils import Progbar
import scipy.io
from termcolor import colored 
import keras.backend as K
import time
import matplotlib.pyplot as plt  

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

def get_weight_grad(model):
    """ Gets gradient of model for given inputs and outputs for all weights"""
    grads = model.optimizer.get_gradients(model.total_loss, model.trainable_weights)
    symb_inputs = (model._feed_inputs + model._feed_targets + model._feed_sample_weights)
    f = K.function(symb_inputs, grads)
    return f

def eval_weight_norm(model,f,inputs,outputs):
    x, y, sample_weight = model._standardize_user_data(inputs, outputs)
    output_grad = f(x + y + sample_weight)
    """output_norm = []
    for grad in output_grad:
        output_norm.append(np.linalg.norm(grad))
    output_norm = np.stack(output_norm,axis=0)"""
    summed_squared = [np.sum(np.square(grad)) for grad in output_grad]
    output_norm = np.sqrt(np.sum(summed_squared))
    return output_norm

def train_SW_batch(data_params,network_params,training_params,real_data,data_path,result_path):
    start_time = time.time()
    batch_size = network_params['batch_size']
    D_rounds = training_params['D_round']
    G_rounds = training_params['G_round']
    begin_epoch = training_params['begin_epoch']
    end_epoch = training_params['end_epoch']
    noverlap = data_params['noverlap']
    nperseg = data_params['nperseg']
    decay = training_params['decay']
    G_lr = training_params['G_lr']
    D_lr =  training_params['D_lr']

    Gen_loss = []
    Dis_loss = []
    Dreal_loss = []
    Dfake_loss = []
    Idx = []
    MMD_list=[]
    RMSE_list = []

    D_grad = []
    D_grad_epoch = []
    D_grad_batch = []
    G_grad = []
    G_grad_epoch = []
    G_grad_batch = []
    Discriminator.trainable = True
    D_f = get_weight_grad(Discriminator)
    Combined.layers[2].trainable = False
    G_f = get_weight_grad(Combined)

    Training_log_file = 'training_log_'+str(begin_epoch)+'_'+str(end_epoch)+'.txt'
    if not os.path.exists(result_path):
        os.mkdir(result_path)

    get_STFT_moment(data_params,real_data,data_path,0,5,result_path,noverlap=noverlap)

    with open(os.path.join(result_path,Training_log_file),'w') as filehandle:
        filehandle.write(str(data_params)+'\n'+str(network_params)+'\n'+str(training_params)+'\n')

    random.shuffle(real_data)
    test_size = int(0.1*data_params['data_size'])
    train_size = data_params['data_size']-test_size
    test_data = real_data[:test_size]
    train_data = list(set(real_data)-set(test_data))
    test_signal = get_batch_signal(data_params,test_data,data_path,0,test_size)
    data_stft = STFT_2C(data_params,train_data,data_path,0,train_size,result_path,noverlap=noverlap)

    nb_batch = int(np.ceil((data_params['data_size']-test_size)/float(network_params['batch_size'])))

    for epoch in range(begin_epoch,end_epoch+1):

        np.random.shuffle(train_data)
        curr_log = 'Epoch ' +str(epoch)+': ' + '\n'
        print(colored('Training epoch '+str(epoch),'yellow'))

        # Learning rate decayed every 50 epochs
        if epoch%(training_params['decay_epoch']) == 0:
            G_lr = G_lr*decay
            D_lr = D_lr*decay
            K.set_value(Combined.optimizer.lr, G_lr)
            K.set_value(Discriminator.optimizer.lr, D_lr)
            curr_log = curr_log + 'G learning rate decayed to '+ str(G_lr) +'\n'
            curr_log = curr_log + 'D learning rate decayed to '+ str(D_lr) +'\n'         
        
        
        for index in range(0,nb_batch-D_rounds,D_rounds):
            if epoch==begin_epoch or True:#float(Dis_loss[-1]) > 0.7*float(Gen_loss[-1]):
                for d in range(D_rounds):
                    batch_stft = STFT_2C(data_params,train_data,data_path,(index+d)*batch_size,(index+d+1)*batch_size,result_path,noverlap=noverlap)
                    batch_real_label = np.ones((batch_size,1))
                    
                    batch_noise = np.random.uniform(-1,1,size=[batch_size,network_params['latent_dim']])
                    batch_generated = Generator.predict(batch_noise)
                    batch_fake_label = np.zeros((batch_size,1))

                    ####### D gradient begin ######
                    evaluated_gradients_real = eval_weight_norm(Discriminator,D_f,batch_stft,batch_real_label)
                    evaluated_gradients_fake = eval_weight_norm(Discriminator,D_f,batch_generated,batch_fake_label)
                    D_grad.append((evaluated_gradients_real+evaluated_gradients_fake)/2)
                    D_grad_epoch.append(epoch)
                    D_grad_batch.append(index)
                    ####### D gradient end ######
                
                    d_b_loss_fake = Discriminator.train_on_batch(batch_generated,batch_fake_label)
                    d_b_loss_real = Discriminator.train_on_batch(batch_stft,batch_real_label)
                    d_b_loss = (d_b_loss_fake + d_b_loss_real)/2
                    print(colored('The D loss at batch %s of epoch %s is %0.6f'%(index+d,epoch,d_b_loss),'blue'))
                
            if epoch==begin_epoch or True:#float(Gen_loss[-1]) > 0.7*float(Dis_loss[-1]):
                for _ in range(G_rounds):
                    batch_noise = np.random.uniform(-1,1,size=[batch_size,network_params['latent_dim']])
                    batch_trick = np.ones((batch_size,1))

                    ####### G gradient begin ######
                    evaluated_gradients = eval_weight_norm(Combined,G_f,batch_noise,batch_trick)                    
                    G_grad.append(evaluated_gradients)
                    G_grad_epoch.append(epoch)
                    G_grad_batch.append(index)
                    ####### G gradient end ######

                    g_b_loss = Combined.train_on_batch(batch_noise,batch_trick)
                    print('The G loss at batch %s of epoch %s is %0.6f'%(index+d,epoch,g_b_loss))

                
        input_noise = np.random.uniform(-1,1,size=(train_size,network_params['latent_dim']))
        generated_stft = Generator.predict(input_noise)

        # Evaluate G loss and D loss
        Idx.append(epoch)
        input_trick = np.ones((train_size,1))
        Gen_loss.append(Combined.evaluate(input_noise,input_trick))
        curr_log = curr_log + '   Generator loss:' + str(Gen_loss[-1]) + '\n'

        # Evaluate D loss in real and fake dataset respectively
        Dreal_loss.append(Discriminator.evaluate(data_stft,np.ones((train_size,1))))
        Dfake_loss.append(Discriminator.evaluate(generated_stft,np.zeros((train_size,1))))

        Dis_loss.append((Dreal_loss[-1]+Dfake_loss[-1])/2)
        curr_log = curr_log + '   Discriminator loss:' + str(Dis_loss[-1]) + '\n'
        print('now Gen_loss is %0.6f'%Gen_loss[-1])
        print('now Dis_loss is %0.6f'%Dis_loss[-1])
        
        input_noise = np.random.uniform(-1,1,size=(train_size,network_params['latent_dim']))
        generated_stft = Generator.predict(input_noise)
        # Evaluate MMD & RMSE
        np.random.shuffle(generated_stft)
        generated_stft_MMD = generated_stft[:test_size,:,:,:]
        generated_signal_MMD = ISTFT_2C(data_params,generated_stft_MMD,result_path,noverlap=noverlap)
        sigma = [pairwisedistances(test_signal[:],generated_signal_MMD[:])]
        mmd = MMD(test_signal[:],generated_signal_MMD[:],sigma)
        rmse = RMSE(np.squeeze(test_signal,axis=-1),np.squeeze(generated_signal_MMD,axis=-1))
        MMD_list.append(mmd)
        RMSE_list.append(rmse)
        curr_log = curr_log + '   MMD:' + str(mmd) + '\n'

        if epoch%10 == 0:
            plt.plot(Idx,MMD_list,'y',label='MMD')
            plt.ylim(0,0.2)
            plt.xlabel('Epoch')
            plt.ylabel('MMD')

            plot_name = 'MMD.png'

            plt.savefig(os.path.join(result_path,plot_name))
            plt.close()

            plt.plot(Idx,RMSE_list,'m')
            plt.xlabel('Epoch')
            plt.ylabel('RMSE')

            plot_name = 'RMSE.png'

            plt.savefig(os.path.join(result_path,plot_name))
            plt.close()

        # Draw loss curves after every 10 epochs
        if epoch%50 == 0:
            plotdata = {'eval_idx': Idx, 'Dreal_loss':Dreal_loss, 'Dfake_loss':Dfake_loss}
            draw_D_loss(plotdata,result_path,epoch)
            plotdata_GD = {'G_loss':Gen_loss,'D_loss':Dis_loss,'G_idx':Idx,'D_idx':Idx}
            draw_train_loss(plotdata_GD,result_path,epoch)
            curr_log = curr_log + 'Loss curve generated'+'\n'
		
        # Save model weights every 10 epochs
        if epoch%50 == 0:
            Gen_wfile = 'GW_epoch_'+str(epoch)+'.hdf5'
            Dis_wfile = 'DW_epoch_'+str(epoch)+'.hdf5'
            Generator.save_weights(os.path.join(result_path,Gen_wfile), True)
            Discriminator.save_weights(os.path.join(result_path,Dis_wfile), True)

		# Plot generated examples every 10 epochs
        if epoch%50 == 0:
            generated_signal = ISTFT_2C(data_params,generated_stft[1:6,:,:,:],result_path,noverlap=noverlap)
            draw_generated_signal(np.array(generated_signal),result_path,epoch=epoch)
            curr_log = curr_log + '   Generated example saved'+'\n'

        if epoch%10 == 0:
            scipy.io.savemat(os.path.join(result_path,'Training_history.mat'),{'G_loss':Gen_loss, 'D_loss':Dis_loss, 'index': Idx, 'MMD':MMD_list, 'RMSE':RMSE_list})
            scipy.io.savemat(os.path.join(result_path,'Gradients.mat'),{'D_grad':D_grad,'D_epoch':D_grad_epoch,'D_batch':D_grad_batch,
                'G_grad':G_grad,'G_epoch':G_grad_epoch,'G_batch':G_grad_batch})

		# Write log file
        with open(os.path.join(result_path,Training_log_file),'a') as filehandle:
            filehandle.write(curr_log)
            filehandle.write('Training time: '+ str((time.time()-start_time)/60) +' min \n')
            filehandle.write('\n')
            filehandle.close()
    
    





if __name__ == "__main__":
    

    data_params = {'data_len':8192,'nb_channel':1,'data_size':1698,
                        'nperseg':256,'noverlap':192,'ntime':128,'nfreq':128}
    network_params = {'latent_dim':100,'dropout_rate':0,'batch_size':64,'dcgan_loss':'lsgan','use_batchnorm':False,
                        'use_MBdisc':True,'MD_nb_kernels':15,'MD_kernel_dim':40}
    training_params = {'D_lr':0.00001, 'G_lr':0.0002, 'decay':0.707,'decay_epoch':100,
                        'D_round':1, 'G_round':10, 'begin_epoch':1, 'end_epoch':2000}
    
    D_opt = Adam(lr=training_params['D_lr'],beta_1=0.5)
    G_opt = Adam(lr=training_params['G_lr'],beta_1=0.5)

    Generator = build_generator(data_params,network_params)
    print(Generator.summary())

    Discriminator = build_discriminator(data_params,network_params)
    if network_params['dcgan_loss'] == 'lsgan':
        Discriminator.compile(
            optimizer=D_opt,
            loss=sigmoid_crossentropy,
        )
    elif network_params['dcgan_loss']=='wgan':
        Discriminator.compile(
            optimizer=D_opt,
            loss=wassertein_loss,
        )
    print(Discriminator.summary())

    noise = Input(shape=(network_params['latent_dim'],))
    generated = Generator(noise)
    fake = Discriminator(generated)
    Discriminator.trainable = False

    Combined = Model(inputs=noise,outputs=fake)
    if network_params['dcgan_loss'] == 'lsgan':
        Combined.compile(
            optimizer=G_opt,
            loss=sigmoid_crossentropy,
        )
    elif network_params['dcgan_loss']=='wgan':
        Combined.compile(
            optimizer=G_opt,
            loss=wassertein_loss,
        )
    print(Combined.summary())
    Discriminator.save_weights('Initial_DW.hdf5',True)
    Combined.save_weights('Initial_GW.hdf5',True)


    data_path = '../NIH_Clean_Sw'
    file_list = []
    for file in os.listdir(data_path):
        file_list.append(os.path.splitext(file)[0])
    data_params['data_size'] = len(file_list)


    curr_path = 'MBDisc_2C_decay_D%dG%d_TOB_Adamlr%0.7fAdam%0.7f'%(
        training_params['D_round'],
        training_params['G_round'],
        training_params['D_lr'],
        training_params['G_lr'])
    """Discriminator.load_weights(os.path.join(curr_path,'DW_epoch_2000.hdf5'))
    Generator.load_weights(os.path.join(curr_path,'GW_epoch_2000.hdf5'))
    Combined.layers[2].set_weights(Discriminator.get_weights())
    Combined.layers[1].set_weights(Generator.get_weights())"""
    
    train_SW_batch(data_params,network_params,training_params,real_data=file_list,data_path=data_path,result_path=curr_path)