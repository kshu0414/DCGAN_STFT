import numpy as np
from utils import get_batch_signal
from scipy.signal import stft, istft
import scipy.io
import os
import math
from matplotlib import pyplot as plt

def get_STFT_moment(data_params,data_list,data_path,start_index,end_index,path,nperseg=256,noverlap=128):
    data_size = end_index-start_index
    ntime = int(np.ceil(data_params['data_len']/(nperseg-noverlap)))
    data_stft = np.zeros((data_size,int(nperseg/2+1),1,ntime))

    for idx in range(start_index,end_index):
        x = get_batch_signal(data_params,data_list,data_path,idx,idx+1)
        x = np.squeeze(x,axis=0)
        _,_,Zxx = stft(x,window='hann',nperseg=nperseg,noverlap=noverlap,return_onesided=True,boundary='zeros',padded=True,axis=0)
        data_stft[idx-start_index,:,:,:] = np.abs(Zxx[:,:,:-1])

    print(np.max(data_stft))

    # Conversion in log scale to align with human perception
    data_stft = np.log(data_stft+1e-6)

    # Normalization with zero mean and unit deviation
    file_name = 'STFTparameters'
    SMean = np.mean(data_stft,axis=(0,3))
    SStd = np.std(data_stft,axis=(0,3),ddof=0)
    scipy.io.savemat(os.path.join(path,file_name+'.mat'),{'SMean':SMean,'SStd':SStd})

def STFT(data_params,data_list,data_path,start_index,end_index,path,nperseg=256,noverlap=128):
    data_size = end_index-start_index
    ntime = int(np.ceil(data_params['data_len']/(nperseg-noverlap)))
    data_stft = np.zeros((data_size,int(nperseg/2+1),1,ntime))

    for idx in range(start_index,end_index):
        x = get_batch_signal(data_params,data_list,data_path,idx,idx+1)
        x = np.squeeze(x,axis=0)
        _,_,Zxx = stft(x,window='hann',nperseg=nperseg,noverlap=noverlap,return_onesided=True,boundary='zeros',padded=True,axis=0)
        data_stft[idx-start_index,:,:,:] = np.abs(Zxx[:,:,:-1])

    # Conversion in log scale to align with human perception
    data_stft = np.log(data_stft+1e-6)

    # Normalization with zero mean and unit deviation
    file_name = 'STFTparameters'
    if not os.path.isfile(os.path.join(path,file_name)):
        SMean = np.mean(data_stft,axis=(0,3))
        SStd = np.std(data_stft,axis=(0,3),ddof=0)
        scipy.io.savemat(os.path.join(path,file_name),{'SMean':SMean,'SStd':SStd})
    else: # For the testing set
        STFT_params = scipy.io.loadmat(os.path.join(path,file_name))
        SMean = STFT_params['SMean']
        SStd = STFT_params['SStd']

    SMean = np.repeat(np.expand_dims(SMean,axis=-1),np.shape(data_stft)[-1],axis=-1)
    SStd = np.repeat(np.expand_dims(SStd,axis=-1),np.shape(data_stft)[-1],axis=-1)
    data_stft = (data_stft-SMean)/SStd/3

    # Bound clipping and rescaling to [-1,1]
    data_stft = np.clip(data_stft,-1,1)
    
    # Permute dimension to [data_size,128(time),128(freq bins),1]
    data_stft = np.transpose(data_stft[:,:-1,:,:],axes=(0,3,1,2))
    return data_stft

def ISTFT(data_params,generated_stft,path,nperseg=256,noverlap=128):
    nfreq = int(nperseg/2)
    ntime = int(np.ceil(data_params['data_len']/(nperseg-noverlap)))
    nstride = nperseg-noverlap
    file_name = 'STFTparameters'
    STFT_params = scipy.io.loadmat(os.path.join(path,file_name))
    SMean = STFT_params['SMean']
    SStd = STFT_params['SStd']

    # Permute and add last frequency bin back
    generated_size = np.shape(generated_stft)[0]
    recons_generated = np.transpose(generated_stft,axes=(0,2,3,1))
    recons_generated = np.hstack((recons_generated,np.zeros((generated_size,1,1,ntime))))

    # Revert normalization
    recons_generated = recons_generated * 3
    SStd_repeat = np.repeat(np.expand_dims(SStd,axis=-1),ntime,axis=-1)
    SMean_repeat = np.repeat(np.expand_dims(SMean,axis=-1),ntime,axis=-1)
    recons_generated = np.multiply(recons_generated,SStd_repeat) + SMean_repeat

    # Log to linear
    recons_generated = np.exp(recons_generated)
    recons_generated = np.concatenate((recons_generated,np.zeros((generated_size,nfreq+1,1,1))),axis=3)

    audio = []
    ntime = np.shape(recons_generated)[-1]
    for idx in range(generated_size):
        S = np.dstack((np.zeros((nfreq+1,1,100)),recons_generated[idx,:,:,:]))
        curr = griffinLims(S,16,nperseg,noverlap)
        curr = curr[nstride*100:nstride*100+data_params['data_len']]
        audio.append(curr)
    
    return audio

def griffinLims(S,maxIter,nfft,Overlap):
    S0 = S.astype('complex128')

    for _ in range(maxIter):
        _,x = istft(S,window='hann',nperseg=nfft,noverlap=Overlap,input_onesided=True,boundary=True,time_axis=-1,freq_axis=-3)
        x = np.transpose(x)
        _,_,S_est = stft(x,window='hann',nperseg=nfft,noverlap=Overlap,return_onesided=True,boundary='zeros',padded=True,axis=0)
        S_est_pos = np.abs(S_est)
        S_est_pos[S_est_pos<1e-6] = 1e-6
        phase = S_est/S_est_pos
        S = np.multiply(phase,S0)
    
    _,x = istft(S,window='hann',nperseg=nfft,noverlap=Overlap,input_onesided=True,boundary=True,time_axis=-1,freq_axis=-3) 
    x = np.transpose(x)
    x = np.real(x)
    return x

def STFT_2C(data_params,data_list,data_path,start_index,end_index,path,nperseg=256,noverlap=128):
    data_size = end_index-start_index
    ntime = int(np.ceil(data_params['data_len']/(nperseg-noverlap)))
    stft_pha = np.zeros((data_size,int(nperseg/2+1),1,ntime))
    stft_mag = np.zeros((data_size,int(nperseg/2+1),1,ntime))

    for idx in range(start_index,end_index):
        x = get_batch_signal(data_params,data_list,data_path,idx,idx+1)
        x = np.squeeze(x,axis=0)
        _,_,Zxx = stft(x,window='hann',nperseg=nperseg,noverlap=noverlap,return_onesided=True,boundary='zeros',padded=True,axis=0)
        stft_mag[idx-start_index,:,:,:] = np.abs(Zxx[:,:,:-1])
        stft_pha[idx-start_index,:,:,:] = np.angle(Zxx[:,:,:-1])/math.pi

    # Conversion in log scale to align with human perception
    stft_mag = np.log(stft_mag+1e-6)

    # Normalization with zero mean and unit deviation
    file_name = 'STFTparameters'
    if not os.path.isfile(os.path.join(path,file_name)):
        SMean = np.mean(stft_mag,axis=(0,3))
        SStd = np.std(stft_mag,axis=(0,3),ddof=0)
        scipy.io.savemat(os.path.join(path,file_name+'.mat'),{'SMean':SMean,'SStd':SStd})
    else: # For the testing set
        STFT_params = scipy.io.loadmat(os.path.join(path,file_name))
        SMean = STFT_params['SMean']
        SStd = STFT_params['SStd']

    SMean = np.repeat(np.expand_dims(SMean,axis=-1),np.shape(stft_mag)[-1],axis=-1)
    SStd = np.repeat(np.expand_dims(SStd,axis=-1),np.shape(stft_mag)[-1],axis=-1)
    stft_mag = (stft_mag-SMean)/SStd/3

    # Bound clipping and rescaling to [-1,1]
    stft_mag = np.clip(stft_mag,-1,1)
    
    # Permute dimension to [data_size,128(time),128(freq bins),1]
    data_stft = np.concatenate((stft_mag,stft_pha),axis=-2)
    data_stft = np.transpose(data_stft[:,:-1,:,:],axes=(0,3,1,2))
    return data_stft

def ISTFT_2C(data_params,generated_stft,path,nperseg=256,noverlap=128):
    nfreq = int(nperseg/2)
    ntime = int(np.ceil(data_params['data_len']/(nperseg-noverlap)))
    nstride = nperseg-noverlap
    file_name = 'STFTparameters'
    STFT_params = scipy.io.loadmat(os.path.join(path,file_name+'.mat'))
    SMean = STFT_params['SMean']
    SStd = STFT_params['SStd']
    generated_size = np.shape(generated_stft)[0]
    audio = np.zeros((generated_size,data_params['data_len'],1))

    # Permute and add last frequency bin back
    recons_generated = np.transpose(generated_stft,axes=(0,2,3,1))
    recons_generated = np.hstack((recons_generated,np.zeros((generated_size,1,2,ntime))))
    stft_mag = np.expand_dims(recons_generated[:,:,0,:],axis=-2)
    stft_pha = np.expand_dims(recons_generated[:,:,1,:],axis=-2)

    # Revert normalization
    stft_mag = stft_mag * 3
    SStd_repeat = np.repeat(np.expand_dims(SStd,axis=-1),ntime,axis=-1)
    SMean_repeat = np.repeat(np.expand_dims(SMean,axis=-1),ntime,axis=-1)
    stft_mag = np.multiply(stft_mag,SStd_repeat) + SMean_repeat

    # Log to linear
    stft_mag = np.exp(stft_mag)
    recons_Zxx = stft_mag * np.exp(1j*stft_pha*math.pi)
    recons_Zxx = np.concatenate((recons_Zxx,np.zeros((generated_size,nfreq+1,1,1))),axis=3)
    for idx in range(generated_size):
        curr_Zxx = np.dstack((np.zeros((nfreq+1,1,100)),recons_Zxx[idx,:,:,:]))
        _,curr_audio = istft(curr_Zxx,window='hann',nperseg=nperseg,noverlap=noverlap,input_onesided=True,boundary=True,time_axis=-1,freq_axis=-3)
        curr_audio = curr_audio[:,nstride*100:nstride*100+data_params['data_len']]
        audio[idx,:,:] = np.transpose(curr_audio)
    
    audio = np.real(audio)
    audio = audio/np.max(audio)

    return audio

def drawSTFTSpec(plotdata,path,epoch):
    data1 = plotdata[0,:]
    data1 = np.transpose(data1)
    data2 = plotdata[1,:]
    data2 = np.transpose(data2)
    data3 = plotdata[2,:]
    data3 = np.transpose(data3)
    data4 = plotdata[3,:]
    data4 = np.transpose(data4)
    data5 = plotdata[4,:]
    data5 = np.transpose(data5)
    tr = 128
    fr = 128

    plt.subplot(321)
    plt.pcolormesh(np.arange(fr),np.arange(tr),data1)
    plt.subplot(322)
    plt.pcolormesh(np.arange(fr),np.arange(tr),data2)
    plt.subplot(323)
    plt.pcolormesh(np.arange(fr),np.arange(tr),data3)
    plt.subplot(324)
    plt.pcolormesh(np.arange(fr),np.arange(tr),data4)
    plt.subplot(325)
    plt.pcolormesh(np.arange(fr),np.arange(tr),data5)
    plt.colorbar()
    plt.suptitle('generated STFT spectrums after '+str(epoch)+' epoch')

    plotname = 'generated_stft_phase'+str(epoch)+'_epoch.png'
    plt.savefig(os.path.join(path,plotname))
    plt.close()

    