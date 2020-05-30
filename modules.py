#!/usr/bin/env python
# coding: utf-8

# In[56]:


import numpy as np 
import h5py
import matplotlib.pyplot as plt
import scipy
import collections
from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap
from hdbscan import HDBSCAN
from scipy import signal, interpolate
from itertools import product
from scipy.signal import firwin, lfilter
from math import *
import matplotlib.font_manager
import matplotlib as mpl
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import matplotlib.cm as cm
from sklearn.metrics import adjusted_rand_score
from scipy.spatial import distance
import GPy
import sys
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm


# In[60]:


def fourier_transform(waveform, num_harmonics, len_spw):
    step=np.arange(0,len_spw,1)
    a=[]
    b=[]
    for order in range(1, num_harmonics + 1):
        w_cos = waveform * np.cos(order * 2 * np.pi / len_spw * step) 
        w_sin = waveform * np.sin(order * 2 * np.pi / len_spw * step)
        a.append((2/len_spw)*w_cos.sum())
        b.append((2/len_spw)*w_sin.sum())

    return [np.array(a),np.array(b)] 


def delt(base,target):
    delt = target - base
    if np.abs(delt) > np.pi:
        if base >= 0:
            delt = 2*pi + delt
        else:
            delt = delt - 2*pi
            
    return delt

import numpy.matlib
def sampling_timing(init_site, width_extended = 100, width_detected = 32, step_width = 0.01, len_wave = 20):
    shift_quantity  = np.round(0.01*np.arange(-99,100),2)
    time_coordinate = np.arange(0,len_wave,step_width) + init_site - width_detected + shift_quantity.reshape(-1,1)#x
    time_coordinate = np.round(100*time_coordinate,0).astype('int')
    
    integer_idx = np.where(time_coordinate%100==0,True, False)
    temp_idx = np.where(integer_idx==True)[1].reshape(199,len_wave)
    samp_idx = time_coordinate[integer_idx].reshape(integer_idx.shape[0],len_wave).copy()
    samp_idx = np.round(samp_idx/100,0).astype('int')
    return samp_idx, temp_idx


class Modules:
    
    def __init__(self):
        self.raw_data_ =  []
        
        self.filtered_data_     = []
        
        self.detected_waveform_ = []
        self.thr = None
        
        self.single_peak_waveform_ = []
        self.multi_peak_waveform_  = []
        
        self.sw_idx_ = []
        self.mw_idx_ = []
    
    def load_data(self, h5_name):
        # Loading a h5 file 
        h5_data = h5py.File(h5_name)
        a_group_key = list(h5_data.keys())[0]
        self.raw_data_ = h5_data[a_group_key][:,0]
        
    def filtering(self, bottom=300, sampling_rate=24000):
        nyq = sampling_rate / 2
        cutoff = np.array([bottom]) / nyq
        numtaps = 255
        bpf = firwin(numtaps, cutoff, pass_zero=False)
        self.filtered_data_ = lfilter(bpf, 1, self.raw_data_)[int((numtaps-1)/2):]
    
    def spike_detection(self, f_std=3.5):
        wide = 36
        self.thr_ = f_std*np.median(np.abs(self.filtered_data_)/0.6745)
        peak_idx = scipy.signal.argrelmax(self.filtered_data_,order=5)[0]
        peak_idx = peak_idx[np.where(self.filtered_data_[peak_idx]>self.thr_)[0]]
        self.detected_waveform_ = [self.filtered_data_[i-wide:i+wide] for i in peak_idx]
        self.detected_waveform_ = np.array(self.detected_waveform_).reshape(peak_idx.shape[0],wide*2)
        self.peak_idx_ =peak_idx.copy()
        
    def dataset_division(self):
        single_peak_waveform = []; multi_peak_waveform  = []; sw_idx = []; mw_idx = []
        for i in range(self.detected_waveform_.shape[0]):
            peak_idx = scipy.signal.argrelmax(self.detected_waveform_[i,:])[0]
            ith_waveform = self.detected_waveform_[i,:].copy()
            if len(np.where(ith_waveform[peak_idx]>self.thr_)[0])==1:
                single_peak_waveform.append(ith_waveform[4:68].copy())
                sw_idx.append(i)
            else:
                multi_peak_waveform.append(ith_waveform[4:68].copy())
                mw_idx.append(i)
        
        self.sw_idx_ = np.array(sw_idx)
        self.mw_idx_ = np.array(mw_idx)
        self.single_peak_waveform_ = np.array(single_peak_waveform).reshape(self.sw_idx_.shape[0],64).copy()
        self.multi_peak_waveform_  = np.array(multi_peak_waveform).reshape(self.mw_idx_.shape[0],64).copy()
     
    
    def feature_extraction(self, diff_order):
        num_spw = self.single_peak_waveform_.shape[0]
        len_spw = self.single_peak_waveform_.shape[1]
        
        self.num_spw_ = num_spw
        self.len_spw_ = len_spw
        
        num_harmonics = int(self.single_peak_waveform_.shape[1]/2)
        self.order_diff_ = diff_order
        self.padded_diff_waveform = np.zeros((num_spw,len_spw))
        
        diff_waveform = np.diff(self.single_peak_waveform_, diff_order).copy()*signal.hamming(len_spw-diff_order)
        self.padded_diff_waveform[:,0:len_spw-diff_order] = np.copy(diff_waveform.T - diff_waveform.mean(1)).T
         
        self.fourier_coeff_a = np.zeros((num_spw, num_harmonics))
        self.fourier_coeff_b = np.zeros((num_spw, num_harmonics))
        self.phase_ = np.zeros((num_spw,num_harmonics))
        self.amplitude_ = np.zeros((num_spw,num_harmonics))
        
        for i in range(num_spw):
            self.fourier_coeff_a[i,:], self.fourier_coeff_b[i,:] = fourier_transform(self.padded_diff_waveform[i,:].copy(),num_harmonics, len_spw)
            self.phase_[i,:] = np.arctan2(self.fourier_coeff_b[i,:],self.fourier_coeff_a[i,:])
            self.amplitude_[i,:] = np.sqrt(self.fourier_coeff_a[i,:]**2+self.fourier_coeff_b[i,:]**2)

            
    def spw_clustering(self,order_harmonics,min_class_size=100):
        self.order_harmonics_ = order_harmonics-1
        hdb = HDBSCAN(min_cluster_size=min_class_size, cluster_selection_method='eom', allow_single_cluster=True).fit(self.amplitude_[:,[self.order_harmonics_]])
        self.classification_labels_ = hdb.labels_.copy()
    
        for label_num in range(int(self.classification_labels_.max())+1):
            target_idx   = np.where(self.classification_labels_==label_num)[0]
            target_phase = self.phase_[target_idx,self.order_harmonics_].copy()

            rpd = [delt(target_phase[i],target_phase[j]) for i in range(target_idx.shape[0]) for j in range(target_idx.shape[0])]
            rpd = np.array(rpd).reshape([target_idx.shape[0],target_idx.shape[0]])
            #misalignment = np.copy(rpd.mean(1)/(2*np.pi*(self.order_harmonics_+1)/self.len_spw_))

            out_idx = target_idx[np.where(np.abs(rpd).mean(1) >= (2*np.pi*(self.order_harmonics_+1)/self.len_spw_))[0]]
            self.classification_labels_[out_idx]= -1    
            #self.rpd_ = rpd.mean(1)
            
    def template_reconstruction(self,fs=0.1):
        
        temp_taxis = np.arange(0,self.len_spw_-self.order_diff_,0.01)
        orig_taxis = np.arange(0,self.len_spw_-self.order_diff_,1)   
        
        template = np.zeros((int(self.classification_labels_.max())+1,temp_taxis.shape[0])) 
        mis = np.zeros(self.classification_labels_.shape[0])
        for label_num in range(int(self.classification_labels_.max())+1):
            target_idx = np.where(self.classification_labels_==label_num)[0]
            mu  = self.amplitude_[target_idx,self.order_harmonics_].mean()
            sig = self.amplitude_[target_idx,self.order_harmonics_].std()
            
            selected_idx = target_idx[np.where((self.amplitude_[target_idx,self.order_harmonics_]>mu-fs*sig)&(self.amplitude_[target_idx,self.order_harmonics_]<mu+fs*sig))[0]]
            selected_waveform = np.diff(self.single_peak_waveform_[selected_idx,:], self.order_diff_).copy()
            selected_waveform = (selected_waveform.T - selected_waveform.mean(1)).T
            selected_phase = self.phase_[selected_idx,self.order_harmonics_].copy()
            
            
            rpd = [delt(selected_phase[i],selected_phase[j]) for i in range(selected_idx.shape[0]) for j in range(selected_idx.shape[0])]
            rpd = np.array(rpd).reshape([selected_idx.shape[0],selected_idx.shape[0]])
            misalignment = np.copy(rpd.mean(1)/(2*np.pi*(self.order_harmonics_+1)/self.len_spw_))
            rpd_m=rpd.mean(1).copy()
            
            rpd_a = []
            for i in range(selected_phase.shape[0]):
                rpd_a.append(delt(rpd_m[i],-selected_phase[i]))
            rpd_a = -np.array(rpd_a)
            
            rpdp = [delt(self.phase_[target_idx[i],self.order_harmonics_],rpd_a.mean()) for i in range(target_idx.shape[0])]
            rpdp = np.array(rpdp)
            
            mis[target_idx] = np.copy(rpdp/(2*np.pi*(self.order_harmonics_+1)/self.len_spw_))
            
            
            x = np.array([orig_taxis[j]+misalignment[i] for i in range(selected_idx.shape[0]) for j in range(self.len_spw_-self.order_diff_) ])
            y = np.array([selected_waveform[i,j]  for i in range(selected_idx.shape[0]) for j in range(self.len_spw_-self.order_diff_) ])  
        
            kernel = GPy.kern.RBF(1,variance=1,lengthscale=1)
            model = GPy.models.GPRegression(x[:,None],y[:,None], kernel=kernel)
            model.optimize()
            [y_pred,var] = model.predict(temp_taxis[:,None])
            template[label_num,:] = y_pred[:,0].copy()
            
        self.template_ = template.copy()
        #self.template_ = np.loadtxt('temp.txt')
            #plt.plot(x,y,marker='o',lw=0)
            #plt.plot(temp_taxis,self.template_[label_num,:])
            #plt.show()
        self.misalignment_ = mis.copy()
        
    def template_matching(self):
        num_temp = self.template_.shape[0]
        template = self.template_[:,1700:3700]
        
        
        shift_quantity  = np.round(0.01*np.arange(-99,100),2)
        dlt = [np.array([i,j]) for i in range(199) for j in range(199)]
        dlt = np.array(dlt,int).reshape((199*199,2))
        tcl = [np.array([i,j]) for i in range(num_temp) for j in range(num_temp)]   
        tcl = np.array(tcl,int).reshape((num_temp*num_temp,2))

        template_tar = [template[i,:] for i in tcl[:,0]]
        template_tar = np.array(template_tar).reshape((tcl.shape[0],template.shape[1]))

        template_adj = [template[i,:] for i in tcl[:,1]]
        template_adj = np.array(template_adj).reshape((tcl.shape[0],template.shape[1]))
        tm_feature = []
        len_ds_temp =20
        
        sh=17
        
        zone=100
        count=0
        shift=[]
        for i_mw in tqdm(self.mw_idx_):
            flag = 0
            
            extd_wave = self.filtered_data_[self.peak_idx_[i_mw]-zone:self.peak_idx_[i_mw]+zone].copy()
            diff_wave = np.diff(extd_wave,self.order_diff_).copy()

            detected_pks = scipy.signal.argrelmax(self.detected_waveform_[i_mw,:])[0]
            detected_pks = detected_pks[np.where(self.detected_waveform_[i_mw,detected_pks]>self.thr_)[0]] - 36 + 100

            extd_pks = scipy.signal.argrelmax(extd_wave)[0]
            extd_pks = extd_pks[np.where(extd_wave[extd_pks]>self.thr_)[0]]
            extd_pks = np.intersect1d(extd_pks, detected_pks)

            init_site = extd_pks
            samp_taxis_tar,temp_taxis_tar = sampling_timing(100+sh)
            idx_adj=init_site[np.argsort(np.abs(init_site-100))[1]]
            
            init_site += sh 
            idx_adj+= sh
            #print(init_site,idx_adj)
            if idx_adj > (100+sh):
                flag=1
                score = np.zeros((dlt.shape[0],tcl.shape[0]))
                samp_taxis_adj, temp_taxis_adj = sampling_timing(idx_adj)  
                for d in range(dlt.shape[0]):
                    dlt_tar, dlt_adj = dlt[d,:]
                    ovlp_idx=np.where(samp_taxis_tar[dlt_tar,:]==samp_taxis_adj[dlt_adj,0])[0]
                    if len(ovlp_idx)==0:
                        x=np.hstack((samp_taxis_tar[dlt_tar,:], samp_taxis_adj[dlt_adj,:]))
                    else:    
                        x=np.hstack((samp_taxis_tar[dlt_tar,0:int(ovlp_idx)], samp_taxis_adj[dlt_adj,:]))
                    wave = diff_wave[x].copy()

                    temp = np.zeros((tcl.shape[0],x.shape[0]))
                    temp[:, 0:temp_taxis_tar.shape[1]] += template_tar[:, temp_taxis_tar[dlt_tar,:]]
                    if len(ovlp_idx)==0:
                        temp[:, temp_taxis_tar.shape[1]:] += template_adj[:, temp_taxis_adj[dlt_adj,:]]
                    else:
                        temp[:, int(ovlp_idx):]           += template_adj[:, temp_taxis_adj[dlt_adj,:]]

                    score[d,:] = np.abs(temp-wave)[:,0:len_ds_temp].mean(1)
                    #score[d,:] = np.abs(temp-wave)[:,17:37].mean(1)

            elif idx_adj < (100+sh):
                flag=-1
                score = np.zeros((dlt.shape[0],tcl.shape[0]))
                samp_taxis_adj, temp_taxis_adj = sampling_timing(idx_adj)        
                for d in range(dlt.shape[0]):
                    dlt_tar, dlt_adj = dlt[d,:]
                    ovlp_idx=np.where(samp_taxis_adj[dlt_adj,:]==samp_taxis_tar[dlt_tar,0])[0]
                    if len(ovlp_idx)==0:
                        x=np.hstack((samp_taxis_adj[dlt_adj,:], samp_taxis_tar[dlt_tar,:]))
                    else:
                        x=np.hstack((samp_taxis_adj[dlt_adj,0:int(ovlp_idx)], samp_taxis_tar[dlt_tar,:]))
                    wave = diff_wave[x].copy()

                    temp = np.zeros((tcl.shape[0],x.shape[0]))
                    temp[:, 0:temp_taxis_adj.shape[1]] += template_adj[:, temp_taxis_adj[dlt_adj,:]]
                    if len(ovlp_idx)==0:
                        temp[:, temp_taxis_adj.shape[1]:]  += template_tar[:, temp_taxis_tar[dlt_tar,:]]
                    else:
                        temp[:, int(ovlp_idx):]            += template_tar[:, temp_taxis_tar[dlt_tar,:]]
                    #print(np.abs(temp-wave)[:,int(ovlp_idx):].mean(1))
                    #print(temp.shape)
                    
                    score[d,:] = np.abs(temp-wave)[:,x.shape[0]-len_ds_temp:].mean(1)
                    
                
            optim_idx = np.unravel_index(np.argmin(score),score.shape)
            
            sub_score = np.zeros((199,3))  
            for dlt_tar in range(199):
                wave = diff_wave[samp_taxis_tar[dlt_tar,:]].copy()
                temp = template[:, temp_taxis_tar[dlt_tar,:]].copy()
                sub_score[dlt_tar,:] = np.abs(temp-wave).mean(1)
            sub_optim_idx = np.unravel_index(np.argmin(sub_score),sub_score.shape)
            
            if sub_score[sub_optim_idx]<score[optim_idx]:
                for n in range(num_temp):
                    tm_feature.append(sub_score[:,n].min())
                
                #fig = plt.figure(dpi=200)
                #plt.plot(np.arange(0,len_ds_temp,0.01)+117-32+shift_quantity[sub_optim_idx[0]],template[sub_optim_idx[1],:] )
                #plt.plot(diff_wave,lw=0.1,marker='.',markersize=5)
                #plt.xlim(50,125)
                #plt.show()
                #print(count,sub_score[sub_optim_idx],shift_quantity[sub_optim_idx[0]],sub_optim_idx[1])
                shift.append(shift_quantity[sub_optim_idx[0]])
                
            else: 
                for n in range(num_temp):
                    tm_feature.append(score[:,n*num_temp:(n+1)*num_temp].min())
            
                #fig = plt.figure(dpi=200)
                #plt.plot(np.arange(0,len_ds_temp,0.01)+117-32+shift_quantity[dlt[optim_idx[0],0]],template[tcl[optim_idx[1],0],:] )
                #plt.plot(np.arange(0,len_ds_temp,0.01)+idx_adj-32+shift_quantity[dlt[optim_idx[0],1]],template[tcl[optim_idx[1],1],:] )
                #plt.plot(diff_wave,lw=0.1,marker='.',markersize=5)
                #plt.xlim(50,125)
                #plt.show()
                #print(count,score[optim_idx],shift_quantity[dlt[optim_idx[0],:]],tcl[optim_idx[1],:])
                shift.append(shift_quantity[dlt[optim_idx[0],0]])
               
            
            count+=1
        self.mpw_shift_ = np.array(shift).copy()
        self.tm_feature_ = np.array(tm_feature).reshape((self.mw_idx_.shape[0],num_temp)).copy()
        #self.tm_feature_=np.load('tmf.txt')
        
        
    def mpw_label_assignment(self,min_csize=10,algo_type='eom'):
        
        HDB = HDBSCAN(min_cluster_size=min_csize,cluster_selection_method=algo_type,allow_single_cluster=True).fit(self.tm_feature_)
        Label_multi=100*np.ones(HDB.labels_.shape[0])
        cl=np.where(HDB.labels_==-1)[0]
        if len(cl)>0:
            Label_multi[cl] = -1
            
        for i in range(int(HDB.labels_.max())+1):
            cl=np.where(HDB.labels_==i)[0]
            Label_multi[cl] = scipy.stats.mode(self.tm_feature_.argmin(1)[cl])[0][0]
        
        self.label_mpw_ = Label_multi.astype('int').copy()
        data=self.tm_feature_.copy()
        pca = PCA(n_components=3)
        data_norm=data-data.mean(0)
        pca.fit(data_norm)
        transformed = pca.fit_transform(data_norm)
        self.transformed_=transformed.copy()
    
    def data_alignment(self):
        class_label = 100*np.ones(self.detected_waveform_.shape[0])
        class_label[self.sw_idx_] = self.classification_labels_.copy()
        class_label[self.mw_idx_] = self.label_mpw_.copy()
        class_label = class_label.astype('int')

        misa = np.zeros(self.detected_waveform_.shape[0])
        misa[self.sw_idx_] = self.misalignment_.copy()
        misa[self.mw_idx_] = -self.mpw_shift_.copy()
        
        self.class_label_ = class_label.copy()
        self.misa_ = misa.copy()

    
    def plot_step1(self):
        fig = plt.figure(dpi=100, figsize=(20,3))
        plt.rcParams['font.family'] ='arial'
        plt.rcParams['font.size'] = 20
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.tick_params(labelright=False, labeltop=False)
        plt.xlim(0,2000)
        plt.xlabel("Time (step)")
        plt.ylabel("Signal")

        plt.plot(self.raw_data_,label='raw data',c='gray',alpha=0.7)
        plt.plot(self.filtered_data_,label='filtered data',alpha=0.7)

        plt.legend(loc = 'best',ncol=2)
        plt.show()
        
    def plot_step2(self):
        fig = plt.figure(dpi=100, figsize=(25,5))
        fig.subplots_adjust(wspace=0.4)
        plt.rcParams['font.family'] ='arial'
        ax1 = fig.add_subplot(1, 4, 1)
        ax2 = fig.add_subplot(1, 4, 2)
        ax3 = fig.add_subplot(1, 4, 3)
        ax4 = fig.add_subplot(1, 4, 4)

        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.tick_params(labelright=False, labeltop=False)
        ax1.set_xlabel('Time (step)')
        ax1.set_ylabel('Signal')
        ax1.set_ylim(self.detected_waveform_.min()*1.05,self.detected_waveform_.max()*1.05)
        ax1.plot(self.detected_waveform_[:,4:68].T,c='gray',alpha=0.1)
        ax1.text(-2,self.detected_waveform_.max()*1.1,r'Detected waveforms (%d)'%self.detected_waveform_.shape[0])

        ax2.annotate('', xy=(1, 0.5), xytext=(-0.25, 0.5), arrowprops=dict(facecolor='black', shrink=0.1))
        ax2.text(0.25,0.55,'Division')
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.spines['bottom'].set_visible(False)
        ax2.spines['left'].set_visible(False)
        ax2.tick_params(labelright=False, labeltop=False,labelleft=False,labelbottom=False, left=False, bottom=False)

        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)
        ax3.tick_params(labelright=False, labeltop=False)
        ax3.set_xlabel('Time (step)')
        ax3.set_ylabel('Signal')
        ax3.set_ylim(self.detected_waveform_.min()*1.05,self.detected_waveform_.max()*1.05)
        ax3.plot(self.single_peak_waveform_.T,c='gray',alpha=0.1)
        ax3.text(-2,self.detected_waveform_.max()*1.1,r'Single-peak waveforms (%d)'%self.single_peak_waveform_.shape[0])

        ax4.spines['top'].set_visible(False)
        ax4.spines['right'].set_visible(False)
        ax4.tick_params(labelright=False, labeltop=False)
        ax4.set_xlabel('Time (step)')
        ax4.set_ylabel('Signal')
        ax4.set_ylim(self.detected_waveform_.min()*1.05,self.detected_waveform_.max()*1.05)
        ax4.plot(self.multi_peak_waveform_.T,c='gray',alpha=0.1)
        ax4.text(-2,self.detected_waveform_.max()*1.1,r'Multi-peak waveforms (%d)'%self.multi_peak_waveform_.shape[0])
        plt.show()

    def plot_step3(self):
        fig = plt.figure(figsize=(25,17),dpi=100)
        plt.rcParams['font.family'] ='arial'
        for i in range(31):
            ax = fig.add_subplot(4, 8, i+1, projection="polar")
            ax.scatter(self.phase_[:,i], self.amplitude_[:,i], marker='.', c='darkgray', s = 10, alpha =0.1)
            ax.set_thetagrids(np.arange(0,360,90),labels=[], fmt=None)
            ax.set_ylim([0, self.amplitude_[:,i].max()*1.05])
            ax.set_rgrids(np.arange(0,self.amplitude_[:,i].max(),self.amplitude_[:,i].max()), labels=[], angle=None)
            ax.set_title(r'%gHz ($O_{\rm h}$=%d)'%((i+1)/(64/24000),i+1),fontsize=15)

    def plot_step4(self):
        fig = plt.figure(figsize=(25,5),dpi=100)
        plt.rcParams['font.family'] ='arial'
        plt.rcParams['font.size'] = 15
        plt.subplots_adjust(wspace=-0.2)
        cmap = plt.get_cmap("tab10")

        ax = fig.add_subplot(1, 2, 1, projection="polar")
        cl = np.where(self.classification_labels_==-1)[0]
        ax.scatter(self.phase_[cl,self.order_harmonics_], self.amplitude_[cl,self.order_harmonics_], marker='x', c='black', s = 10, alpha = 0.5)
        cl = np.where(self.classification_labels_!=-1)[0]
        ax.scatter(self.phase_[cl,self.order_harmonics_], self.amplitude_[cl,self.order_harmonics_], marker='o', c=cmap(self.classification_labels_[(cl)]), s = 10, alpha = 0.5)
        ax.set_thetagrids(np.arange(0,360,90),labels=[], fmt=None)
        ax.set_ylim([0, self.amplitude_[:,self.order_harmonics_].max()*1.05])
        ax.set_rgrids(np.arange(0,self.amplitude_[:,self.order_harmonics_].max(),self.amplitude_[:,self.order_harmonics_].max()), labels=[], angle=None)
        ax.set_title(r'$O_{\rm d}$ = %d, $O_{\rm h}$ = %d'%(self.order_diff_, self.order_harmonics_+1),fontsize=15)


        ax = fig.add_subplot(1, 2, 2)

        for i in range(self.classification_labels_.max()+1):
            cl=np.where(self.classification_labels_==i)[0]
            x=np.arange(0,64-self.order_diff_,1)+self.misalignment_[cl].reshape(-1,1)
            dw = np.diff(self.single_peak_waveform_[cl,:],self.order_diff_).copy() 
            if i==0:
                ax.plot(x[0,:],dw[0,:],marker='.',markersize=2,lw=0,c=cmap(i),alpha=0.2,label=r'%d$^{\rm th}$ order difference of single-peak waveform (shifted)'%self.order_diff_)
                ax.plot(x[1:,:].T,dw[1:,:].T,marker='.',markersize=2,lw=0,c=cmap(i),alpha=0.2)
            else:
                ax.plot(x.T,dw.T,marker='.',markersize=2,lw=0,c=cmap(i),alpha=0.2)

        for i in range(self.classification_labels_.max()+1):
            if i==0:
                ax.plot(np.arange(0,64-self.order_diff_,0.01),self.template_[i,:],c='black',label='Template for each class')
            else:
                ax.plot(np.arange(0,64-self.order_diff_,0.01),self.template_[i,:],c='black')

        ax.set_xlim(17,37)
        ax.set_xticks(np.arange(17,38,1))
        ax.set_xlabel('Time (step)')
        ax.set_ylabel('Signal')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(labelright=False, labeltop=False)
        lgnd = ax.legend(bbox_to_anchor=(1, 1.2),ncol=2,numpoints=5,loc='upper right')
        lgnd.legendHandles[0]._legmarker.set_color('black')
        lgnd.legendHandles[0]._legmarker.set_alpha(0.5)
       
        plt.show()
        
        
    def plot_step5(self):
        fig = plt.figure(figsize=(25,5),dpi=100)
        plt.rcParams['font.family'] ='arial'
        plt.rcParams['font.size'] = 15
        plt.subplots_adjust(wspace=-0.2)
        cmap = plt.get_cmap("tab10")
        
        ax = fig.add_subplot(1, 2, 1)
        cl=np.where(self.label_mpw_==-1)[0]
        ax.scatter(self.transformed_[cl,0],self.transformed_[cl,1],c='black',alpha=0.5,marker='x',s=10)
        cl=np.where(self.label_mpw_!=-1)[0]
        ax.scatter(self.transformed_[cl,0],self.transformed_[cl,1],c=cmap(self.label_mpw_[cl]),s=10,alpha=0.5,marker='o')
        ax.set_xlim(self.transformed_[cl,:].min()*1.1,self.transformed_[cl,:].max()*1.1)
        ax.set_ylim(self.transformed_[cl,:].min()*1.1,self.transformed_[cl,:].max()*1.1)
        ax.set_aspect('equal',adjustable='box')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(labelright=False, labeltop=False,labelbottom=False,labelleft=False,bottom=False, left=False)

        ax = fig.add_subplot(1, 2, 2)
        for i in range(int(self.label_mpw_.max())+1):
            cl=np.where(self.label_mpw_==i)[0]
            x=np.arange(0,64-self.order_diff_,1)-self.mpw_shift_[cl].reshape(-1,1)
            dw = np.diff(self.multi_peak_waveform_[cl,:],self.order_diff_).copy()
            if i==0:
                ax.plot(x[0,:],dw[0,:],marker='o',markersize=3,lw=0,c=cmap(i),alpha=0.4,label=r'%d$^{\rm th}$ order difference of multi-peak waveform (shifted)'%self.order_diff_)
                ax.plot(x[1:,:].T,dw[1:,:].T,marker='o',markersize=3,lw=0,c=cmap(i),alpha=0.4)
            else:
                ax.plot(x.T,dw.T,marker='o',markersize=3,lw=0,c=cmap(i),alpha=0.4)
        for i in range(int(self.label_mpw_.max())+1):
            if i==0:
                ax.plot(np.arange(0,64-self.order_diff_,0.01),self.template_[i,:],c='black', label='Template constructed form single-peak waveforms')
            else:
                ax.plot(np.arange(0,64-self.order_diff_,0.01),self.template_[i,:],c='black')

        ax.set_xlim(17,37)
        ax.set_xticks(np.arange(17,38,1))
        ax.set_xlabel('Time (step)')
        ax.set_ylabel('Signal')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        lgnd = ax.legend(bbox_to_anchor=(1, 1.2),ncol=2,numpoints=5,loc='upper right')
        lgnd.legendHandles[0]._legmarker.set_color('black')
        lgnd.legendHandles[0]._legmarker.set_alpha(0.3)
        lgnd.legendHandles[0]._legmarker.set_markersize(2.5)
        plt.show()

    def plot_step6(self):
        fig = plt.figure(figsize=(25,5),dpi=100)
        plt.rcParams['font.family'] ='arial'
        plt.rcParams['font.size'] = 20
        plt.subplots_adjust(wspace=0.3)
        cmap = plt.get_cmap("tab10")

        ymax = self.detected_waveform_[:,4:68].max()*1.1
        ymin = self.detected_waveform_[:,4:68].min()*1.1

        if np.where(self.class_label_==-1)[0].shape[0]>0:
            num = self.class_label_.max()+2
            flag=1
        else:
            num = self.class_label_.max()+1
            flag=0

        for i in range(1,num):
            ax = fig.add_subplot(1, num, i)
            cl=np.where(self.class_label_==i-1)[0]
            ax.plot((np.arange(0,64)+self.misa_[cl].reshape(-1,1)).T,self.detected_waveform_[cl,4:68].T,c=cmap(i-1),marker='.',lw=0,alpha=0.1)
            ax.set_xlim(0,64)
            ax.set_ylim(ymin,ymax)
            ax.set_xlabel('Time (step)')
            ax.set_ylabel('Signal')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.text(0,ymax*1.1,'Class #%d (%d waveforms)'%(i,cl.shape[0]))

        if flag==1:
            ax = fig.add_subplot(1, num, num)
            cl=np.where(self.class_label_==-1)[0]
            ax.plot(self.detected_waveform_[cl,4:68].T,c='black',marker='.',lw=0,alpha=0.1)
            ax.set_xlim(0,64)
            ax.set_ylim(ymin,ymax)
            ax.set_xlabel('Time (step)')
            ax.set_ylabel('Signal')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.text(0,ymax*1.1,'Outlier (%d waveforms)'%cl.shape[0])

        plt.show()






# In[ ]:





# In[ ]:




