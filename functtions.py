#!/usr/bin/env python
# coding: utf-8

# In[398]:


### ========================================================

__all__ = ['autocorr', 'buildPlots', 'buildClusters', 'calc_kl', 'calc_MI2', 'cellFunction','cluster_mxn', 'create_cnn_ae', 'create_stacked_ae', 'create_stacked_sae', 'createPath', 'custom_loss', 'dataAnalysis', 'derivCellFunction', 'derivXtalkFunction','DateInfo', 'deNorm', 'extract_e', 'extractToSave','extraConfigs', 'fileClose', 'fileOpenClose', 'fileOpenWrite', 'findFile', 'genTau_0', 'genCellSamples', 'genXTcSamples', 'genSampDelay', 'genNoise', 'getSigmClus','getSigmCells','getBest', 'getMinEtaPhi', 'getOccurences','getData', 'getDictParam', 'genRandVal','getErange', 'getMeanRms', 'getIdxSampClus','getIdxClus_mxn', 'getSize', 'latexTableAnalysis', 'loadEbinData', 'loadEbinDict','loadData', 'loadSaveDict', 'loadSignals', 'min_max', 'modelConfigs', 'modelEvaluate', 'modelPredict', 'NNmodel', 'OF_coeffs','OptFilt', 'OptimalFilter', 'plot_heatmap', 'plot_mse', 'plot_mse', 'plot_RE_distribution', 'plot_rmse', 'plot_training_error', 'plotClusters','plotBar', 'plotBoxplot', 'plotClusters', 'plotHist2columns', 'plotHisto', 'plotLoss', 'plotScatter', 'plotSignal', 'printInfo', 'r2score', 'readDirectCells', 'recEnergyTimeNoNoise', 'regModel', 'rms', 'rms_e', 'rmseLoss', 'rmsErr', 'plotSigmClus', 'plotSigmCells', 'ShowerShapes','sigmaTauLoss', 'sizeFile', 'sortArray', 'tableTrain', 'timeInfo', 'wContActFunc', 'writeSigm', 'wEndContReScore', 'wInitContReScore','XTalk', 'topText', 'middleText', 'bottomText']

import os
import tempfile
os.environ["MPLCONFIGDIR"] = tempfile.gettempdir()

#from memory_profiler import profile
import copy
import sys

import psutil
from collections import Counter

from pandas import DataFrame
from datetime import datetime, timedelta, timezone

from itertools import permutations, combinations

from copy import deepcopy
from math import atan, sinh, cosh, tanh, exp, log, sqrt, sin, cos, gamma
#import keras
import pickle as pkl
import tensorflow.keras
from tensorflow.keras import backend as K, optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras import layers, regularizers, models
from tensorflow.keras.layers import Activation, Dense, Conv2D, Dropout, BatchNormalization, Input, Reshape, Flatten, Conv2DTranspose, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model, Sequential
#from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras.metrics import RootMeanSquaredError as rmse
from tensorflow.keras.metrics import MeanSquaredLogarithmicError as msle
from scipy.stats import gmean
#import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import cm
from matplotlib.gridspec import GridSpec
import pylab as plt
import glob
import random
import math 
import gc

from multiprocessing import Pool, cpu_count

import numpy as np
from numpy import array, heaviside

import pandas as pd
import pathlib
import pickle
from scipy.misc import derivative
from scipy.stats import ks_2samp

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mutual_info_score, r2_score, mean_squared_error as mse

from sklearn import preprocessing
from sklearn.preprocessing import normalize as norm
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import time
from XTconstants import *
from tensorflow.python.keras.layers import pooling

###################################
## Plot definitions
###################################

SMALL_SIZE  = 10
MEDIUM_SIZE = 16  # 15
LARGE_SIZE  = 18  # 14
STICK_SIZE  = 14  # 14
BIGGER_SIZE = 20  # 12
HUGE_SIZE   = 22
#plt.rcParams.update({
#  "text.usetex": True,
#  "font.family": "Palatino"
#})
### rcParams['font.family'] = 'sans-serif'
### rcParams['font.sans-serif'] = ['Tahoma', 'DejaVu Sans',
###                               'Lucida Grande', 'Verdana']
### font.family        : serif
### font.serif         : Times, Palatino, New Century Schoolbook, Bookman, Computer Modern Roman
### font.sans-serif    : Helvetica, Avant Garde, Computer Modern Sans Serif
### font.cursive       : Zapf Chancery
### font.monospace     : Courier, Computer Modern Typewriter

### text.usetex        : true
###
plt.rcParams["figure.figsize"] = (10,6)
plt.rcParams.update({'figure.max_open_warning': 0})
plt.rc('font',   size      = HUGE_SIZE)  # controls default text sizes
plt.rc('axes',   titlesize = HUGE_SIZE)   # fontsize of the axes title
plt.rc('axes',   labelsize = HUGE_SIZE)  # fontsize of the x and y labels
plt.rc('xtick',  labelsize = STICK_SIZE)   # fontsize of the tick labels
plt.rc('ytick',  labelsize = STICK_SIZE)   # fontsize of the tick labels
plt.rc('legend', fontsize  = BIGGER_SIZE)  # legend fontsize
plt.rc('figure', titlesize = BIGGER_SIZE)  # fontsize of the figure title

##########################################
Ndpi = 500

        
## ==========================================================
class buildClusters:
    
    def __init__(self, rawEtruth, rawET, rawEta, rawPhi):
        self.ClusAmplitudes = self.ClusAmplitudes()
        self.ClusSamples    = self.ClusSamples()
        self.nClus      = len( rawEtruth )
        self.thrsd      = 1000
        self.rawE       = rawEtruth
        self.rawET      = rawET
        self.rawEta     = rawEta
        self.rawPhi     = rawPhi
        self.cellsDelay = [0.03354785, 0.0548546 , 0.02556418, 0.07777289, 0.05689534, 0.06669829, 0.04451098, 
                           0.09679919, 0.0342638 , 0.00694144, 0.06594262, 0.06816116, 0.01071184, 0.03591564, 
                           0.02394185, 0.00638659, 0.06317729, 0.07036964, 0.07063568, 0.07236611, 0.03953515, 
                           0.02027446, 0.05412756, 0.03588347, 0.0452883 , 0.09808412, 0.05593064, 0.00394952, 
                           0.00444025, 0.01181117, 0.04472722, 0.01380569, 0.04275083, 0.06421954, 0.07084368,
                           0.0367995 , 0.0493678 , 0.03964693, 0.00665569, 0.01898337, 0.04562767, 0.02710452, 
                           0.03433142, 0.0867104 , 0.03017023, 0.04573883, 0.06316867, 0.09278989, 0.0406065 ]
        
    #def get_EetaPhiXTclus7x7(rawEtruth, rawET, rawEta, rawPhi):
    def clus_7x7_Amplitudes(self):
        # Define the size of the cluster (m x n) around the maximum value
        m = 7  # Number of rows
        n = 7  # Number of columns

        E_7x7, ET_7x7, eta_7x7, phi_7x7, noise, XTc_Clus, XTl_Clus, XTr_Clus = [], [], [], [], [], [], [], []
        start = time.time()
        #print(f'Extracting clusters ...',end='\r', flush=True)
        for idx, rawE in enumerate(self.rawE):
            
            nEta = len(dict(Counter( self.rawPhi[idx] )))
            nPhi = len(dict(Counter( self.rawEta[idx] )))

            df_E   = pd.DataFrame(rawE.reshape(nPhi,nEta))
            df_ET  = pd.DataFrame(self.rawET[idx].reshape(nPhi,nEta))
            df_eta = pd.DataFrame(self.rawEta[idx].reshape(nPhi,nEta))
            df_phi = pd.DataFrame(self.rawPhi[idx].reshape(nPhi,nEta))

            # Find the location (row and column labels) of the maximum value
            max_value_location = df_E.stack().idxmax()
            max_row, max_col = max_value_location

            # Get the column labels as a list
            column_labels = list(df_E.columns)

            # Calculate the row and column boundaries for the cluster
            start_row = df_E.index.get_loc(max_row) - m // 2
            end_row   = start_row + m
            start_col = column_labels.index(max_col) - n // 2
            end_col   = start_col + n

            # Ensure that the boundaries are within the DataFrame's limits
            start_row = max(0, start_row)
            end_row   = min(df_E.shape[0], end_row)
            start_col = max(0, start_col)
            end_col   = min(df_E.shape[1], end_col)

            # Extract the clusters based on the boundaries
            newE = df_E.iloc[start_row:end_row, start_col:end_col].to_numpy()
            
            E_7x7.append( newE.flatten() )
            ET_7x7.append( df_ET.iloc[start_row:end_row, start_col:end_col].to_numpy().flatten() )
            eta_7x7.append( df_eta.iloc[start_row:end_row, start_col:end_col].to_numpy().flatten() )
            phi_7x7.append( df_phi.iloc[start_row:end_row, start_col:end_col].to_numpy().flatten() )

            noise.append( genNoise( m*n, norm=True ) )
            
            XTc, XTl, XTr = self.relativeClusAmplitude( newE )
                        
            XTc_Clus.append(XTc.flatten())
            XTl_Clus.append(XTl.flatten())
            XTr_Clus.append(XTr.flatten())
                        
        #XTc_Clus = np.vstack(XTc_Clus).flatten()
        #XTl_Clus = np.vstack(XTl_Clus).flatten()
        #XTr_Clus = np.vstack(XTr_Clus).flatten()
        
        #XTc_Clus, XTl_Clus, XTr_Clus = self.relativeClusAmplitude(np.reshape(E_7x7[0], [7,7]) )    

        self.ClusAmplitudes.E     = array(np.vstack( E_7x7 ))
        self.ClusAmplitudes.ET    = array(np.vstack( ET_7x7 ))
        self.ClusAmplitudes.eta   = array(np.vstack( eta_7x7 ))
        self.ClusAmplitudes.phi   = np.vstack( phi_7x7 )
        self.ClusAmplitudes.XT_C  = array( XTc_Clus )
        
        #self.ClusAmplitudes.XT_C  = np.vstack( E_7x7 )*XTc_Clus
                
        
        self.ClusAmplitudes.XT_L  = array( XTl_Clus )
        self.ClusAmplitudes.XT_R  = array( XTr_Clus )
        self.ClusAmplitudes.Noise = np.vstack( noise )
                
        #print(f' { self.ClusAmplitudes.E.shape[0] } 7x7 clusters extracted! Elapsed time: {timeInfo(time.time()-start)}')
#        return {'Etruth': np.vstack( E_7x7 ), 
#                'ET_truth': np.vstack( ET_7x7 ), 
#                'Eta': np.vstack( eta_7x7 ), 
#                'Phi': np.vstack( phi_7x7 ), 
#                'XTc': np.vstack(E_7x7)*XTc_Clus.flatten(), 
#                'XTl': np.vstack(E_7x7)*XTl_Clus.flatten(), 
#                'XTr': np.vstack(E_7x7)*XTr_Clus.flatten(), 
#                'Noise': np.vstack( noise ) }

    ## =============================================
    def clusSigSamplesParalel(self):
        start = time.time()
        
        ## get all 7x7 clusters
        self.clus_7x7_Amplitudes()
        ## --------------

        num_clusters = len(self.ClusAmplitudes.E)
        
        print('-'*30)
        print(' Parallel mode! '.center(30,'*'))
        print(f'Sampling {num_clusters} 7x7 clusters ...', end='\r', flush=True)
        #dictSignals = self.genClusSignSamples()
                
        #nCores = psutil.cpu_count(logical=False)
        with Pool(10) as pool:
        # Use partial to create a function with fixed arguments
            #process_func = partial(self.genClusSignSampParal, idx)

        # Generate clusters in parallel
            res = pool.map(self.genClusSignSampParal, range(num_clusters))                              
                
        G_samp      = array([res[i][0] for i in range(len(res))] )
        G_PrimeSamp = array([res[i][1] for i in range(len(res))] )
        xtC_samp    = array([res[i][2] for i in range(len(res))] )
        xtL_samp    = array([res[i][3] for i in range(len(res))] )
        xtR_samp    = array([res[i][4] for i in range(len(res))] )
        noise_samp  = array([res[i][5] for i in range(len(res))] )
        
        self.ClusSamples.G      = array([res[i][0] for i in range(len(res))] )
        self.ClusSamples.Gprime = array([res[i][1] for i in range(len(res))] )
        self.ClusSamples.E      = np.repeat( self.ClusAmplitudes.E, 4, axis=1 )*G_samp
        self.ClusSamples.XT_C   = np.repeat( self.ClusAmplitudes.XT_C, 4, axis=1 )*xtC_samp
        self.ClusSamples.XT_L   = np.repeat( self.ClusAmplitudes.XT_L, 4, axis=1 )*xtL_samp
        self.ClusSamples.XT_R   = np.repeat( self.ClusAmplitudes.XT_R, 4, axis=1 )*xtR_samp
        self.ClusSamples.Noise  = noise_samp
        
        print(f'Sampling {num_clusters} 7x7 clusters is done! Elapsed time: {timeInfo(time.time()-start)}')                
        

    ## =============================================
    def clusSigSamples(self):
        start = time.time()
        
        ## get all 7x7 clusters
        self.clus_7x7_Amplitudes()
        ## --------------
        print(('-'*20).center(30,' '))        
        print(f'Sampling {len(self.ClusAmplitudes.E)} 7x7 clusters ...', end='\r', flush=True)
                
        dictSignals = self.genClusSignSamples()
                        
        self.ClusSamples.G      = dictSignals['G_CellsSamp']
        self.ClusSamples.Gprime = dictSignals['G_PrimeCellsSamp']
        self.ClusSamples.E      = np.repeat( self.ClusAmplitudes.E, 4, axis=1 )*dictSignals['G_CellsSamp']
        self.ClusSamples.XT_C   = np.repeat( self.ClusAmplitudes.XT_C, 4, axis=1 )*dictSignals['XTc_samp']
        self.ClusSamples.XT_L   = np.repeat( self.ClusAmplitudes.XT_L, 4, axis=1 )*dictSignals['XTl_samp']
        self.ClusSamples.XT_R   = np.repeat( self.ClusAmplitudes.XT_R, 4, axis=1 )*dictSignals['XTr_samp']
        self.ClusSamples.Noise  = array(dictSignals['NoiseSamp'])
        
        print(f'Sampling {len(self.ClusAmplitudes.E)} 7x7 clusters is done! Elapsed time: {timeInfo(time.time()-start)}')  
    
    ## =============================================
    def genTau_0(self):
        import numpy as np

        tau_0_mean = 0.5        # arbitrary bias to validate method and to avoid singularity on ML training
        tau_std    = 0.5

        return np.random.normal(tau_0_mean,tau_std)

    ## =============================================
    def genSampDelay(self, n=None):
        tDelay = []
        if n is None: n=49 ## for a 7x7 cluster celss

        while len( tDelay )<n:
            data = np.random.normal(0, 0.1)
            if abs(data) > 0.1: pass
            else: tDelay.append( abs(data) ) 

        return np.array( tDelay )

    ## =============================================

    def genClusSignSamples(self, m_size=None, n_size=None ):
        nClus = self.nClus
        
        if m_size is None and n_size is None:
            n_size, m_size = 7, 7

        ## nCells define cluster size
        nCells = m_size*n_size
        G_CellsSamp, G_PrimeCellsSamp, xtC, xtL, xtR, noise = [], [], [], [], [], []

        ## for each cell on cluster on time delay.
        #cellsDelay = genSampDelay( nCells ) 
        
        cellsDelay = self.cellsDelay

        ## Build nClus with 4 samples each. 
        ## Different number of samples should be provided as nSamp= n on genCellSamples
        for _ in range( nClus ):
            ## tau_0 is a single arbitrary time of flight per cluster
            tau_0 = self.genTau_0()

            G_CellsSamp.append( self.genCellSignSamples( cellsDelay, tau_0 )[0] )
            G_PrimeCellsSamp.append( self.genCellSignSamples( cellsDelay, tau_0 )[1] )
            xtC.append( self.genCellSignSamples( cellsDelay, tau_0 )[2] )
            xtL.append( self.genCellSignSamples( cellsDelay, tau_0 )[3] )
            xtR.append( self.genCellSignSamples( cellsDelay, tau_0 )[4] )        
            noise.append( array(self.genNoise( nCells )) )
        
        return {'G_CellsSamp':array( G_CellsSamp ), 
                'G_PrimeCellsSamp': array( G_PrimeCellsSamp ), 
                'XTc_samp': xtC,
                'XTl_samp': xtL,
                'XTr_samp': xtR,
                'NoiseSamp': noise}
    
    ## =============================================

    def genClusSignSampParal(self, idx, m_size=None, n_size=None ):
        #nClus = self.nClus
        
        self.clus_7x7_Amplitudes()
        
        if m_size is None and n_size is None:
            n_size, m_size = 7, 7

        ## nCells define cluster size
        nCells = m_size*n_size
        G_CellsSamp, G_PrimeCellsSamp, xtC, xtL, xtR, noise = [], [], [], [], [], []

        ## for each cell on cluster on time delay.
        #cellsDelay = genSampDelay( nCells ) 
        
        cellsDelay = self.cellsDelay
        
            ## tau_0 is a single arbitrary time of flight per cluster
        tau_0 = self.genTau_0()
        
        return self.genCellSignSamples( cellsDelay, tau_0 )
        
        #return {'G_CellsSamp':array( G_CellsSamp ), 
                #'G_PrimeCellsSamp': array( G_PrimeCellsSamp ), 
                #'XTc_samp': xtC,
                #'XTl_samp': xtL,
                #'XTr_samp': xtR,
                #'NoiseSamp': noise}
    
    ## =============================================
    def genNoise(self, n=None, norm=False):    
        g_AmpNoise    = 50   ## MeV    
        noise = np.zeros(n)
        for i in range(n):
            noise[i] = np.random.normal(0,2)

        return array(g_AmpNoise*(noise/max(noise)))
    
    ## =============================================

    def genCellSignSamples( self, cellsDelay, tau_0, nSamp=None):
        '''
           This function produces the signals samples for a single cell:
           - G Cell
           - G prime Cell
           - XTc
           - XTl
           - XTr
           - Noise
        '''
        if nSamp is None: nSamp = 4

        G_CellsSamp, G_PrimeCellsSamp, xt_C, xt_L, xt_R, noise = [], [], [], [], [], []
        for idx, delay in enumerate(cellsDelay):
            for n in range(nSamp):
                G_CellsSamp.append( cellFunction( 25*(n+1) + delay + tau_0 ))
                G_PrimeCellsSamp.append( derivCellFunction( 25*(n+1) + delay + tau_0 ))

                xt_C.append( XTalk( 25*(n+1) + delay + tau_0) )
                xt_L.append( XTalk( 25*(n+1) + delay + tau_0) )
                xt_R.append( cellFunction( 25*(n+1) + delay + tau_0 ))

        noise.append( array(self.genNoise( len( cellsDelay )) ))
        
        return array( G_CellsSamp ), array( G_PrimeCellsSamp ), array( xt_C ), array( xt_L ), array( xt_R ), array( noise )

    ## =============================================
    def get_EetaPhiXTclus7x7_samp(self, Etruth):
        nSamp = 4
        nClus = len(Etruth)

        vectDelay = genSampDelay(nSamp*i*j)
        g_tau_0   = genTau_0( len(elf.clusEtruth) )

        return

    ## =============================================
    def relativeClusAmplitude(self, Etruth, debug=False):
            ## If debug is true is possible to verify is relative cluster is correct
            ## which means evaluate relationship between first neighbohood with respect
            ## to XTl (8 cells around interest cell) and XTc (4 cells, UP/DOWN, LEFT/RIGHT)
        i, j        = np.shape(Etruth)      
        if debug == True:
            clusXTc  = np.linspace(1, i*j, i*j, dtype=int).reshape(i, j)
            clusXTl  = np.linspace(1, i*j, i*j, dtype=int).reshape(i, j)
            clusXTr  = np.linspace(1, i*j, i*j, dtype=int).reshape(i, j)
        else :
            clusXTc  = Etruth*g_AmpXt_C
            clusXTl  = Etruth*g_AmpXt_L
            clusXTr  = Etruth*g_AmpXt_R

            #clusXTc, clusXTl, clusXTr = np.ones([i,i])*g_AmpXt_C, np.ones([i,i])*g_AmpXt_L, np.ones([i,i])*g_AmpXt_R

        XTc_Cluster = np.zeros(Etruth.shape)
        XTl_Cluster = np.zeros(Etruth.shape)
        XTr_Cluster = np.zeros(Etruth.shape)

        for (phi, eta), _ in np.ndenumerate(Etruth):
                # corner 00
            if phi == 0 and eta == 0:
                XTc_Cluster[phi, eta] = clusXTc[phi+1, eta] + clusXTc[phi, eta+1]
                XTl_Cluster[phi, eta] = clusXTl[phi+1, eta] + clusXTl[phi, eta+1] + clusXTl[phi+1, eta+1]
                XTr_Cluster[phi, eta] = clusXTr[phi+1, eta] + clusXTr[phi, eta+1] + clusXTr[phi+1, eta+1]

                # first row
            if phi == 0 and 0 < eta <  j-1:
                XTc_Cluster[phi, eta] = clusXTc[phi, eta-1] + clusXTc[phi, eta+1] + clusXTc[phi+1, eta]
                XTl_Cluster[phi, eta] = clusXTl[phi, eta-1] + clusXTl[phi, eta+1] + clusXTl[phi+1, eta] + clusXTl[phi+1, eta-1] + clusXTl[phi+1, eta+1]
                XTr_Cluster[phi, eta] = clusXTr[phi, eta-1] + clusXTr[phi, eta+1] + clusXTr[phi+1, eta] + clusXTr[phi+1, eta-1] + clusXTr[phi+1, eta+1]

                # corner 0j
            if eta == j-1 and phi == 0:
                XTc_Cluster[phi, eta] = clusXTc[phi, eta-1] + clusXTc[phi+1, eta]
                XTl_Cluster[phi, eta] = clusXTl[phi, eta-1] + clusXTl[phi+1, eta] + clusXTl[phi+1, eta-1]
                XTr_Cluster[phi, eta] = clusXTr[phi, eta-1] + clusXTr[phi+1, eta] + clusXTr[phi+1, eta-1]            

                # first column
            if eta == 0 and 0 < phi <  i-1:
                XTc_Cluster[phi, eta] = clusXTc[phi-1, eta] + clusXTc[phi+1, eta] + clusXTc[phi, eta+1]
                XTl_Cluster[phi, eta] = clusXTl[phi-1, eta] + clusXTl[phi+1, eta] + clusXTl[phi, eta+1] + clusXTl[phi-1, eta+1] + clusXTl[phi+1, eta+1]
                XTr_Cluster[phi, eta] = clusXTr[phi-1, eta] + clusXTr[phi+1, eta] + clusXTr[phi, eta+1] + clusXTr[phi-1, eta+1] + clusXTr[phi+1, eta+1]

                # center
            if 0 < phi < i-1 and 0 < eta <  j-1:
                #print(f'eta: {eta} | phi: {phi}')
                XTc_Cluster[phi, eta] = clusXTc[phi-1, eta] + clusXTc[phi, eta-1] + clusXTc[phi, eta+1] + clusXTc[phi+1, eta]
                XTl_Cluster[phi, eta] = clusXTl[phi-1, eta] + clusXTl[phi, eta-1] + clusXTl[phi, eta+1] + clusXTl[phi+1, eta] + clusXTl[phi-1, eta+1] + clusXTl[phi-1, eta-1] + clusXTl[phi+1, eta+1] + clusXTl[phi+1, eta-1]
                XTr_Cluster[phi, eta] = clusXTr[phi-1, eta] + clusXTr[phi, eta-1] + clusXTr[phi, eta+1] + clusXTr[phi+1, eta] + clusXTr[phi-1, eta+1] + clusXTr[phi-1, eta-1] + clusXTr[phi+1, eta+1] + clusXTr[phi+1, eta-1]

                # last column
            if eta == j-1 and 0 < phi <  i-1:        
                XTc_Cluster[phi, eta] = clusXTc[phi-1, eta] + clusXTc[phi+1, eta] + clusXTc[phi, eta-1]
                XTl_Cluster[phi, eta] = clusXTl[phi-1, eta] + clusXTl[phi+1, eta] + clusXTl[phi, eta-1] + clusXTl[phi-1, eta-1] + clusXTl[phi+1, eta-1]
                XTr_Cluster[phi, eta] = clusXTr[phi-1, eta] + clusXTr[phi+1, eta] + clusXTr[phi, eta-1] + clusXTr[phi-1, eta-1] + clusXTr[phi+1, eta-1]

                # last row
            if phi == i-1 and 0 < eta <  j-1:
                #print(f'eta: {eta} | phi: {phi}')
                XTc_Cluster[phi, eta] = clusXTc[phi, eta-1] + clusXTc[phi, eta+1] + clusXTc[phi-1, eta]
                XTl_Cluster[phi, eta] = clusXTl[phi, eta-1] + clusXTl[phi, eta+1] + clusXTl[phi-1, eta] + clusXTl[phi-1, eta-1] + clusXTl[phi-1, eta+1]
                XTr_Cluster[phi, eta] = clusXTr[phi, eta-1] + clusXTr[phi, eta+1] + clusXTr[phi-1, eta] + clusXTr[phi-1, eta-1] + clusXTr[phi-1, eta+1]

                # corner i0
            if phi == i-1 and eta == 0:
                XTc_Cluster[phi, eta] = clusXTc[phi-1, eta] + clusXTc[phi, eta+1]
                XTl_Cluster[phi, eta] = clusXTl[phi-1, eta] + clusXTl[phi, eta+1] + clusXTl[phi-1, eta+1]
                XTr_Cluster[phi, eta] = clusXTr[phi-1, eta] + clusXTr[phi, eta+1] + clusXTr[phi-1, eta+1]

                # corner ij
            if phi == i-1 and eta == j-1:
                XTc_Cluster[phi, eta] = clusXTc[phi, eta-1] + clusXTc[phi-1, eta]
                XTl_Cluster[phi, eta] = clusXTl[phi, eta-1] + clusXTl[phi-1, eta] + clusXTl[phi-1, eta-1]
                XTr_Cluster[phi, eta] = clusXTr[phi, eta-1] + clusXTr[phi-1, eta] + clusXTr[phi-1, eta-1]

        return XTc_Cluster, XTl_Cluster, XTr_Cluster
    
    class ClusAmplitudes:
        def __init__(self):
            self.E     = []
            self.ET    = []
            self.XT_C  = []
            self.XT_L  = []
            self.XT_R  = []
            self.Noise = []
            
    class ClusSamples:
        def __init__(self):
            self.E     = []
            self.ET    = []
            self.XT_C  = []
            self.XT_L  = []
            self.XT_R  = []
            self.Noise = []


## ==========================================================
class cluster_mxn:

    def __init__(self, EtEtaPhi, size_eta=None, size_phi=None):
        
        self.ShowerShapes = self.ShowerShapes()
        self.XTsamples =  self.XTsignals()
        self.thrsd     = 1000
        self.EtEtaPhi  = EtEtaPhi
        self.clusXT_C  = []
        self.clusXT_L  = []
        self.clusXT_R  = []
        self.clusNoise = []
        self.setEtaTruth = []
        self.setPhiTruth = []
        self.clusEtruth  = []

        #output  ='/eos/user/m/msandesd/SWAN_projects/lorenzetti'
        output  ='..'

        self.pathOut = createPath(f'{output}/lzt_xt')
        
        if size_eta is None: self.size_eta = 7
        if size_phi is None: self.size_phi = 7
            
#     def InitClusters(self, size_eta=None, size_phi=None, idxShowerShapes=None):
#             ## Initial configurations for the cluster_mxn object
#             ## If size_eta and size_phi are None the standard cluster 7x7 is built
#         #self.SetClusterSize(self, size_eta=self.size_eta, size_phi=self.size_phi, idxShowerShapes=idxShowerShapes)
#         #self.SetClusterSize()
#         self.BuildClusters()
#         self.ComputeXT()
#         self.PrintInfo()
#         self.Reta()
#         self.Rphi()
#         self.Weta2()    
            
            ## This function look for clusters (size_eta x size_phi) with hottest cell at the center
    def ComputeSS(self, thrsd=None):
        if thrsd is not None:
            self.thrsd = thrsd
        
        size_eta, size_phi = 7, 7
        self.idxClus7x7 = self.SetClusterSize( size_eta=size_eta, size_phi=size_phi )
        self.BuildClusters( size_eta=size_eta, size_phi=size_phi )
        nCores = psutil.cpu_count(logical=False)
        with Pool(nCores) as pool:
            res = pool.map(self.ComputeXTmultiCore, range(len(self.idxClus7x7)) )        
            pool.close()
            pool.join()

        self.ShowerShapes.clusE_7x7      = self.clusEtruth
        self.ShowerShapes.clusXTc_7x7    = [res[i][0].reshape(size_eta, size_phi) for i in range(len(res))]
        self.ShowerShapes.clusXTl_7x7    = [res[i][1].reshape(size_eta, size_phi) for i in range(len(res))]
        self.ShowerShapes.clusXTr_7x7    = [res[i][2].reshape(size_eta, size_phi) for i in range(len(res))]
        self.ShowerShapes.clusNoise_7x7  = [res[i][3].reshape(size_eta, size_phi) for i in range(len(res))]

        print(f'Computing Reta SS...', end="\r")
        self.Reta()
        print(f'Computing Reta SS... is done!')
        #print('is done!'.rjust(20))
        print(f'Computing Rphi SS...', end="\r")
        self.Rphi()
        print(f'Computing Rphi SS... is done!')
        #print('is done!'.rjust(20))
        print(f'Computing Weta2 SS...', end="\r")
        self.Weta2()
        #print('is done!'.rjust(20))
        print(f'Computing Weta2 SS... is done!')
        
        size_eta, size_phi = 5, 5            
        outputs = self.GetCluster_mxn(size_eta=size_eta, size_phi=size_phi)         
        
        self.ShowerShapes.clusE_5x5      = outputs[0]
        self.ShowerShapes.clusXTc_5x5    = outputs[1]
        self.ShowerShapes.clusXTl_5x5    = outputs[2]
        self.ShowerShapes.clusXTr_5x5    = outputs[3]
        self.ShowerShapes.clusNoise_5x5  = outputs[4]
        
    def SetClusterSize(self, size_eta=None, size_phi=None):    
        stdClusters, idxClusters, nEtaStd, nPhiStd, etaList, phiList, etaHotList, phiHotList = [], [], [], [], [], [], [], [] 
        flag = 0
        
        if size_eta is None: size_eta = 7            
        #else : self.size_eta = self.size_eta
            
        if size_phi is None: size_phi = 7
        #else : self.size_phi = self.size_phi
            
        if size_eta is None and size_phi is None:
            flag = 1
#         if self.thrsd is None: thrsd = 1500 ## in MeV
#         else :  thrsd = self.thrsd
        thrsd = self.thrsd
        for i, signalsCluster in enumerate(self.EtEtaPhi):          
            energy, nEta, nPhi  = signalsCluster[:,0], len(set(signalsCluster[:,1])), len(set(signalsCluster[:,2]))
            
            if sum(energy) < thrsd: pass
            else :                
                    ## adjust nEta, nPhi to keep cluster information
                if len(energy) > nPhi*nEta:
                    nEta, nPhi = getMinEtaPhi(signalsCluster)
                    ## adjust cluster size filling with zeros with respct to nEta, nPhi
                if len(energy) < nPhi*nEta:
                    while len(energy) < nPhi*nEta:
                        energy = np.append(energy,0)                
                etaMax = int(list(energy).index(max(energy))/nPhi)
                phiMax = list(energy).index(max(energy))%nPhi                

                    ## Select the clusters where it is possible to build a cluster
                    ## with the hottest cell at the center of a cluster standard (size_eta x size_phi)
                    ## or a different size for a SS analysis
                if (etaMax + int(size_eta/2)+1 > nEta) or (etaMax - int(size_eta/2) < 0) or (phiMax + int(size_phi/2)+1 > nPhi) or (phiMax - int(size_phi/2) < 0):
                    pass
                else :
                    idxClusters.append(i)
                    stdClusters.append(signalsCluster)
            #self.idxClusters    = idx
            #self.stdClusters    = stdClusters

        return idxClusters

    def SetClusterSize_first(self, size_eta=None, size_phi=None):
        stdClusters, idxClusters, nEtaStd, nPhiStd, etaList, phiList, etaHotList, phiHotList = [], [], [], [], [], [], [], [] 
        flag = 0
        
        if size_eta is None: size_eta = 7            
        #else : self.size_eta = self.size_eta
            
        if size_phi is None: size_phi = 7
        #else : self.size_phi = self.size_phi
            
        if size_eta is None and size_phi is None:
            flag = 1
                                            
        for i, signalsCluster in enumerate(self.EtEtaPhi):          
            energy, nEta, nPhi  = signalsCluster[:,0], len(set(signalsCluster[:,1])), len(set(signalsCluster[:,2]))

                ## adjust nEta, nPhi to keep cluster information
            if len(energy) > nPhi*nEta:
                nEta, nPhi = getMinEtaPhi(signalsCluster)
                ## adjust cluster size filling with zeros with respct to nEta, nPhi
            if len(energy) < nPhi*nEta:
                while len(energy) < nPhi*nEta:
                    energy = np.append(energy,0)      
            #else :
                ## ----------------------------------------
                ## remove cluters where it wasn't possible to generate a cluster 
                ## nPhi x nEta with the hottest cell at the center
                ## -------------
            etaMax = int(list(energy).index(max(energy))/nPhi)
            phiMax = list(energy).index(max(energy))%nPhi                
#             phiMax = int(list(energy).index(max(energy))/nEta)
#             etaMax = int(list(energy).index(max(energy))%nEta)

                ## Select the clusters where is possible to build a cluster
                ## with hottest cell at the center of a cluster standard (size_eta x size_phi)
                ## or a different size for a SS analysis
            if (etaMax + int(size_eta/2)+1 > nEta) or (etaMax - int(size_eta/2) < 0) or (phiMax + int(size_phi/2)+1 > nPhi) or (phiMax - int(size_phi/2) < 0):
                pass
            else :
                idxClusters.append(i)
                stdClusters.append(signalsCluster)
        #self.idxClusters    = idx
        #self.stdClusters    = stdClusters

        return idxClusters
    
    def PrintInfo(self):
        if self.flagStdCluster == 1:
            print(f'{len(self.idxClusters)} standard clusters (eta, phi) ({self.size_eta} , {self.size_phi}) with the Hottest cell at the center extracted from {len(self.EtEtaPhi)} !')
            print(f'Set a different (size_eta, size_phi) to build a specific cluster size!')
        else :
            print(f'{len(self.idxClusters)} standard clusters (eta, phi) ({self.size_eta}, {self.size_phi}) with the Hottest cell at the center extracted from {len(self.EtEtaPhi)}!\n')

    def BuildClusters(self, size_eta=None, size_phi=None, index=None):
        ## set of clusters (nPhi x nEta) taking into account the hottest cell at the center: self.idx        print('inside buildClusters method')        

        Etruth, EtruthNoNegative, idx, etaTruthClusters, eTruthClusters, phiTruthClusters = [], [], [], [], [], []

        if size_eta is None: size_eta = self.size_eta
        else: size_eta = size_eta

        if size_phi is None: size_phi = self.size_phi
        else: size_phi = size_phi

        print(f'Building clusters: {size_eta}x{size_phi}...', end="\r")
            
            ## To compute SS in the same set of clusters it is necessary to setup 
            ##   the set of cluster using the most restrictive set, wich means
            ##   to use the size of denominator of Reta/Rphi
        if index is not None:
            idxClusters = index
        else :
            idxClusters = self.SetClusterSize(size_eta, size_phi)
            
        for idx in idxClusters:
            
            energy = list( self.EtEtaPhi[idx][:,0] )
            nPhi   = len(set( self.EtEtaPhi[idx][:,2] ))
            nEta   = len(set( self.EtEtaPhi[idx][:,1] ))            
              
            phiList = list(set( self.EtEtaPhi[idx][:,2] ))
            etaList = list(set( self.EtEtaPhi[idx][:,1] ))
            
            phiList.sort()
            etaList.sort()
    
            nPhi   = len(set( phiList ))
            nEta   = len(set( etaList ))
                        
            if len(energy) > nPhi*nEta:
                nEta, nPhi = getMinEtaPhi(self.EtEtaPhi[idx])
                ## adjust cluster size filling with zeros with respct to nEta, nPhi
            if len(energy) < nPhi*nEta: 
                #print(f'inside if, idx: {idx}')
                #pass
                while len(energy) < nPhi*nEta:
                    energy = np.append(energy,0)                                
            #else :
            etaMax = int(list(energy).index(max(energy))/nPhi)
            phiMax = list(energy).index(max(energy))%nPhi

            energy = array(energy).reshape(nEta,nPhi)
            
                ## adjust nEta, nPhi to keep cluster information            

            etaTruthClusters.append( etaList[ etaMax - int(size_eta/2): etaMax + int(size_eta/2)+1 ] )
            phiTruthClusters.append( phiList[ phiMax - int(size_phi/2): phiMax + int(size_phi/2)+1 ] )
            #eTruthClusters.append( energy[ etaMax - int(size_eta/2): etaMax + int(size_eta/2)+1, phiMax - int(size_phi/2): phiMax + int(size_phi/2)+1 ] )
            Eraw = energy[ etaMax - int(size_eta/2): etaMax + int(size_eta/2)+1, phiMax - int(size_phi/2): phiMax + int(size_phi/2)+1 ]
            
            eTruthClusters.append(Eraw)
            eNoNegative = np.zeros(Eraw.shape)
            
        #self.clusEtaTruth = etaTruthClusters        
        self.setEtaTruth = etaTruthClusters
        self.setPhiTruth = phiTruthClusters
        self.clusEtruth  = eTruthClusters        
        #self.clusEnoNegative = EtruthNoNegative
        
        #print('is done!'.rjust(20))
        print(f'Building clusters: {size_eta}x{size_phi}... is done!')
    
    #XTc, XTl, Noise = [],[],[]
        
    def ComputeXT(self, clusEnoNegative=None):
            ## Compute the XT inductive and capacitive to build a relative contributions on each cell on the cluster
        XTc, XTl, XTr, Noise, clusSampXT_C, clusSampXT_L, clusSampNoise, clusSampCell = [], [], [], [], [], [], []
        nSamp = 4
            
        g_tau_0   = genTau_0( len(elf.clusEtruth) )
        for idx, cluster in enumerate(setEclusters):
                ## To add XT contributions it is necessary compute the XT for each cluster
                ## to compute each cluster noise
            Etruth    = cluster
            i, j      = np.shape(Etruth)
            
            vectDelay = genSampDelay(nSamp*i*j)

            #[clusCellSamp, derivClusCellSamp], clusXTcSamp, clusXTlSamp  = genCellSamples(vectDelay, g_tau_0[idx], nSamp), genXTcSamples(vectDelay, g_tau_0[idx], nSamp), genXTlSamples(vectDelay, g_tau_0[idx], nSamp)
            
            clusNoise = genNoise(i*j, norm=True)
            clusNoise = clusNoise.reshape(Etruth.shape)
            
                ## To avoid and exlude clusters 1-D
            if i == 1 or j==1: pass
            else :
                ## Computing XT capacitive and inductive contributions
                ## Given a cell in the clusters its first neighbors (8) add capacitive and inductive contributions
                  ## If a cell is in the position (1,1) the full contribution is:
                    ## Capacitive only cells up-down, left-right
                    ## Inductive all 8 neighbors
                  ## If a cell is in the top/bottom line, left/right column in the cluster, the contribution is partial
                  ## because its neighborhood are less than 8 cells
                    
                XTc_Cluster, XTl_Cluster, XTr_Cluster = self.RelativeClusters(Etruth)
                
                XTc.append(XTc_Cluster)
                XTl.append(XTl_Cluster)
                XTr.append(XTr_Cluster)
                Noise.append(clusNoise)
                
        self.clusXT_C  = XTc
        self.clusXT_L  = XTl
        self.clusXT_R  = XTr
        self.clusNoise = Noise
        
        #return XTc#, XTl, Noise
       
    def ComputeXTmultiCore(self, idx, clusEnoNegative=None):
        """
            To compute XT uxing multicore methods it is mandatory call BuildClusters() to define the set of standard clusters
            1st - call SetClustersSize()
            2nd - call BuildCllusters()
        """            
        XTc, XTl, Noise, clusSampXT_C, clusSampXT_L, clusSampXT_R, clusSampNoise, clusSampCell, derivClusCellSamp = [], [], [], [], [], [], [], [], []
        
        nSamp = 4        
          
        Etruth = self.clusEtruth[idx]
                ## To add XT contributions it is necessary compute the XT for each cluster
                ## to compute each cluster noise
        i, j      = np.shape(Etruth)
        #g_tau_0   = genTau_0(nSamp*i*j)0
        g_tau_0   = genTau_0( len(self.clusEtruth ) )
        #vectDelay = genSampDelay(nSamp*i*j)
        vectDelay = genSampDelay(i*j)

        [clusCellSamp, dervClusCellSamp], clusXTcSamp, clusXTlSamp  = genCellSamples(vectDelay, g_tau_0[idx], nSamp), genXTcSamples(vectDelay, g_tau_0[idx], nSamp), genXTlSamples(vectDelay, g_tau_0[idx], nSamp)

        clusNoise = genNoise(i*j, norm=True)
        clusNoise = clusNoise.reshape(Etruth.shape)
        clusXTrSamp = g_AmpXt_R*clusCellSamp

        ## Computing XT capacitive and inductive contributions
        ## Given a cell in the clusters its first neighbors (8) add capacitive and inductive contributions
          ## If a cell is in the position (1,1) the full contribution is:
            ## Capacitive only cells up-down, left-right
            ## Inductive all 8 neighbors
          ## If a cell is in the top/bottom line, left/right column in the cluster, the contribution is partial
          ## because its neighborhood are less than 8 cells

        XTc_Cluster, XTl_Cluster, XTr_Cluster = self.RelativeClusters(Etruth)

#         XTc.append(XTc_Cluster)
#         XTl.append(XTl_Cluster)
#         Noise.append(clusNoise)  
        
        #return XTc_Cluster.reshape(i*j), XTl_Cluster.reshape(i*j), clusNoise.reshape(i*j), clusCellSamp, derivClusCellSamp
        return XTc_Cluster, XTl_Cluster, XTr_Cluster, clusNoise, clusCellSamp, derivClusCellSamp
        
    def XTclusters5x5(self):        
        clusSampXT_C, clusSampXT_L, clusSampXT_R, clusSampNoise, clusSampCell, derivClusSampCell = [], [], [], [], [], []
        nSamp = 4

        size_eta, size_phi = 5, 5
        idxClusters  = self.SetClusterSize( size_eta=size_eta, size_phi=size_phi )        
        
        self.BuildClusters( size_eta=size_eta, size_phi=size_phi )
        print('Computing XT...', end="\r")
        #self.ComputeXT()
        nCores = psutil.cpu_count(logical=False)
        with Pool(nCores) as pool:
            res = pool.map(self.ComputeXTmultiCore, range(len( idxClusters ) ) )
            pool.close()
            pool.join()

        self.clusE_5x5     = self.clusEtruth
        self.clusXTc_5x5   = [res[i][0] for i in range(len(res))]
        self.clusXTl_5x5   = [res[i][1] for i in range(len(res))]
        self.clusXTr_5x5   = [res[i][2] for i in range(len(res))]
        self.clusNoise_5x5 = [res[i][3] for i in range(len(res))]
        self.clusSampCell  = [res[i][4] for i in range(len(res))]
        self.clusDerivSampCell = [res[i][4] for i in range(len(res))]
        
        g_tau_0   = genTau_0( len(self.clusEtruth) )
        for idx, Etruth in enumerate(self.clusEtruth ):
            i, j      = np.shape(Etruth)
            
            #vectDelay = genSampDelay(nSamp*i*j)
            vectDelay = genSampDelay(i*j)
            clusNoise = genNoise(i*j*nSamp, norm=True)
            #clusNoise = clusNoise.reshape(Etruth.shape)
            
            [clusCellSamp, derivClusCellSamp], clusXTcSamp, clusXTlSamp  = genCellSamples(vectDelay, g_tau_0[idx], nSamp), genXTcSamples(vectDelay, g_tau_0[idx], nSamp), genXTlSamples(vectDelay, g_tau_0[idx], nSamp)
            
            XTc_Cluster, XTl_Cluster, XTr_Cluster = self.RelativeClusters(Etruth)
            
            clusXTrSamp = g_AmpXt_R*clusCellSamp
                    
            clusSampXT_C.append((np.repeat(XTc_Cluster, nSamp, axis=1)*clusXTcSamp.reshape(i, j*nSamp)).reshape(i*j*nSamp) )
            clusSampXT_L.append((np.repeat(XTl_Cluster, nSamp, axis=1)*clusXTlSamp.reshape(i, j*nSamp)).reshape(i*j*nSamp) )
            clusSampXT_R.append((np.repeat(XTr_Cluster, nSamp, axis=1)*clusXTrSamp.reshape(i, j*nSamp)).reshape(i*j*nSamp) )
            #clusSampNoise.append(np.repeat(clusNoise, nSamp, axis=1))
            clusSampNoise.append(clusNoise)
            clusSampCell.append((np.repeat(Etruth, nSamp, axis=1)*clusCellSamp.reshape(i, j*nSamp)).reshape(i*j*nSamp) )
            derivClusSampCell.append((np.repeat(Etruth, nSamp, axis=1)*derivClusCellSamp.reshape(i, j*nSamp)).reshape(i*j*nSamp) )
                
#         self.clusSampXTc_5x5    = clusSampXT_C
#         self.clusSampXTl_5x5    = clusSampXT_L
#         self.clusSampNoise_5x5  = clusSampNoise
#         self.clusSampE_5x5      = clusSampCell
#         self.clusSampDerivE_5x5 = derivClusSampCell
#         self.clusTau_0          = g_tau_0

        self.XTsamples.XTc    = clusSampXT_C
        self.XTsamples.XTl    = clusSampXT_L
        self.XTsamples.XTr    = clusSampXT_R
        self.XTsamples.Noise  = clusSampNoise
        self.XTsamples.E      = clusSampCell
        self.XTsamples.DerivE = derivClusSampCell
        self.XTsamples.clusTau_0  = g_tau_0

        #print('is done!'.format( " "*20))
        print('Computing XT... is done!')
        
        
    def XTclusters7x7(self):        
        clusSampXT_C, clusSampXT_L, clusSampXT_R, clusSampNoise, clusSampCell, derivClusSampCell = [], [], [], [], [], []
        nSamp = 4

        size_eta, size_phi = 7, 7
        idxClusters  = self.SetClusterSize( size_eta=size_eta, size_phi=size_phi )        
        
        self.BuildClusters( size_eta=size_eta, size_phi=size_phi )
        print('Computing XT...', end="\r")
        #self.ComputeXT()
        nCores = psutil.cpu_count(logical=False)
        with Pool(nCores) as pool:
            res = pool.map(self.ComputeXTmultiCore, range(len( idxClusters ) ) )
            pool.close()
            pool.join()

        self.clusE_7x7     = self.clusEtruth
        self.clusXTc_7x7   = [res[i][0] for i in range(len(res))]
        self.clusXTl_7x7   = [res[i][1] for i in range(len(res))]
        self.clusXTr_7x7   = [res[i][2] for i in range(len(res))]
        self.clusNoise_7x7 = [res[i][3] for i in range(len(res))]
        self.clusSampCell  = [res[i][4] for i in range(len(res))]
        self.clusDerivSampCell = [res[i][4] for i in range(len(res))]
        
        g_tau_0   = genTau_0( len(self.clusEtruth) )
        for idx, Etruth in enumerate(self.clusEtruth ):
            i, j      = np.shape(Etruth)
            
            #vectDelay = genSampDelay(nSamp*i*j)
            vectDelay = genSampDelay(i*j)
            clusNoise = genNoise(i*j*nSamp, norm=True)
            #clusNoise = clusNoise.reshape(Etruth.shape)
            
            [clusCellSamp, derivClusCellSamp], clusXTcSamp, clusXTlSamp  = genCellSamples(vectDelay, g_tau_0[idx], nSamp), genXTcSamples(vectDelay, g_tau_0[idx], nSamp), genXTlSamples(vectDelay, g_tau_0[idx], nSamp)
            
            XTc_Cluster, XTl_Cluster, XTr_Cluster = self.RelativeClusters(Etruth)
            
            clusXTrSamp = g_AmpXt_R*clusCellSamp
                    
            clusSampXT_C.append((np.repeat(XTc_Cluster, nSamp, axis=1)*clusXTcSamp.reshape(i, j*nSamp)).reshape(i*j*nSamp) )
            clusSampXT_L.append((np.repeat(XTl_Cluster, nSamp, axis=1)*clusXTlSamp.reshape(i, j*nSamp)).reshape(i*j*nSamp) )
            clusSampXT_R.append((np.repeat(XTr_Cluster, nSamp, axis=1)*clusXTrSamp.reshape(i, j*nSamp)).reshape(i*j*nSamp) )
            #clusSampNoise.append(np.repeat(clusNoise, nSamp, axis=1))
            clusSampNoise.append(clusNoise)
            clusSampCell.append((np.repeat(Etruth, nSamp, axis=1)*clusCellSamp.reshape(i, j*nSamp)).reshape(i*j*nSamp) )
            derivClusSampCell.append((np.repeat(Etruth, nSamp, axis=1)*derivClusCellSamp.reshape(i, j*nSamp)).reshape(i*j*nSamp) )
                
#         self.clusSampXTc_5x5    = clusSampXT_C
#         self.clusSampXTl_5x5    = clusSampXT_L
#         self.clusSampNoise_5x5  = clusSampNoise
#         self.clusSampE_5x5      = clusSampCell
#         self.clusSampDerivE_5x5 = derivClusSampCell
#         self.clusTau_0          = g_tau_0

        self.XTsamples.XTc    = clusSampXT_C
        self.XTsamples.XTl    = clusSampXT_L
        self.XTsamples.XTr    = clusSampXT_R
        self.XTsamples.Noise  = clusSampNoise
        self.XTsamples.E      = clusSampCell
        self.XTsamples.DerivE = derivClusSampCell
        self.XTsamples.clusTau_0  = g_tau_0

        #print('is done!'.format( " "*20))
        print('Computing XT... is done!')
    #dftCluster_mxn(self, size_eta=None, size_phi=None, clusXT_C=None, clusXT_L=None, clus_Noise=None):
    def GetCluster_mxn(self, size_eta=None, size_phi=None):
        
        i,j = self.ShowerShapes.clusE_7x7[0].shape
        etaCenter = int(i/2)
        phiCenter = int(j/2)
         
        #clusXTc, clusXTl, clusNoise = np.zeros([i,j]), np.zeros([i,j]), np.zeros([i,j])
        clusEnergy, clusXTc, clusXTl, clusXTr, clusNoise  = [], [], [], [], []
        
        for i in range(len(self.ShowerShapes.clusE_7x7)):
            clusEnergy.append( self.ShowerShapes.clusE_7x7[i][etaCenter - int(size_eta/2): etaCenter + int(size_eta/2)+1, phiCenter - int(size_phi/2): phiCenter + int(size_phi/2)+1] )
            
            clusXTc.append( self.ShowerShapes.clusXTc_7x7[i][etaCenter - int(size_eta/2): etaCenter + int(size_eta/2)+1, phiCenter - int(size_phi/2): phiCenter + int(size_phi/2)+1] )
            
            clusXTl.append( self.ShowerShapes.clusXTl_7x7[i][etaCenter - int(size_eta/2): etaCenter + int(size_eta/2)+1, phiCenter - int(size_phi/2): phiCenter + int(size_phi/2)+1] )
            
            clusXTr.append( self.ShowerShapes.clusXTr_7x7[i][etaCenter - int(size_eta/2): etaCenter + int(size_eta/2)+1, phiCenter - int(size_phi/2): phiCenter + int(size_phi/2)+1] )
            
            clusNoise.append( self.ShowerShapes.clusNoise_7x7[i][etaCenter - int(size_eta/2): etaCenter + int(size_eta/2)+1, phiCenter - int(size_phi/2): phiCenter + int(size_phi/2)+1] )
        
        return clusEnergy, clusXTc, clusXTl, clusXTr, clusNoise

        ## ----------------------
        ## ----------------------

    def RelativeClusters(self, Etruth, debug=False):
            ## If debug is true is possible to verify is relative cluster is correct
            ## which means evaluate relationship between first neighbohood with respect
            ## to XTl (8 cells around interest cell) and XTc (4 cells, UP/DOWN, LEFT/RIGHT)
        i, j        = np.shape(Etruth)            
        if debug == True:
            clusXTc  = np.linspace(1, i*j, i*j, dtype=int).reshape(i, j)
            clusXTl  = np.linspace(1, i*j, i*j, dtype=int).reshape(i, j)
            clusXTr  = np.linspace(1, i*j, i*j, dtype=int).reshape(i, j)
        else :
            clusXTc  = Etruth*g_AmpXt_C
            clusXTl  = Etruth*g_AmpXt_L
            clusXTr  = Etruth*g_AmpXt_R
        
        XTc_Cluster = np.zeros(Etruth.shape)
        XTl_Cluster = np.zeros(Etruth.shape)
        XTr_Cluster = np.zeros(Etruth.shape)

        for (phi, eta), Et in np.ndenumerate(Etruth):
                # first row
            if phi == 0 and 0 < eta <  j-1:
                XTc_Cluster[phi, eta] = clusXTc[phi, eta-1] + clusXTc[phi, eta+1] + clusXTc[phi+1, eta]
                XTl_Cluster[phi, eta] = clusXTl[phi, eta-1] + clusXTl[phi, eta+1] + clusXTl[phi+1, eta] + clusXTl[phi+1, eta-1] + clusXTl[phi+1, eta+1]
                XTr_Cluster[phi, eta] = clusXTr[phi, eta-1] + clusXTr[phi, eta+1] + clusXTr[phi+1, eta] + clusXTr[phi+1, eta-1] + clusXTr[phi+1, eta+1]
                # last row
            if phi == i-1 and 0 < eta <  j-1:
                #print(f'eta: {eta} | phi: {phi}')
                XTc_Cluster[phi, eta] = clusXTc[phi, eta-1] + clusXTc[phi, eta+1] + clusXTc[phi-1, eta]
                XTl_Cluster[phi, eta] = clusXTl[phi, eta-1] + clusXTl[phi, eta+1] + clusXTl[phi-1, eta] + clusXTl[phi-1, eta-1] + clusXTl[phi-1, eta+1]
                XTr_Cluster[phi, eta] = clusXTr[phi, eta-1] + clusXTr[phi, eta+1] + clusXTr[phi-1, eta] + clusXTr[phi-1, eta-1] + clusXTr[phi-1, eta+1]

                # center
            if 0 < phi < i-1 and 0 < eta <  j-1:
                #print(f'eta: {eta} | phi: {phi}')
                XTc_Cluster[phi, eta] = clusXTc[phi-1, eta] + clusXTc[phi, eta-1] + clusXTc[phi, eta+1] + clusXTc[phi+1, eta]
                XTl_Cluster[phi, eta] = clusXTl[phi-1, eta] + clusXTl[phi, eta-1] + clusXTl[phi, eta+1] + clusXTl[phi+1, eta] + clusXTl[phi-1, eta+1] + clusXTl[phi-1, eta-1] + clusXTl[phi+1, eta+1] + clusXTl[phi+1, eta-1]
                XTr_Cluster[phi, eta] = clusXTr[phi-1, eta] + clusXTr[phi, eta-1] + clusXTr[phi, eta+1] + clusXTr[phi+1, eta] + clusXTr[phi-1, eta+1] + clusXTr[phi-1, eta-1] + clusXTr[phi+1, eta+1] + clusXTr[phi+1, eta-1]

                # first column
            if eta == 0 and 0 < phi <  i-1:
                XTc_Cluster[phi, eta] = clusXTc[phi-1, eta] + clusXTc[phi+1, eta] + clusXTc[phi, eta+1]
                XTl_Cluster[phi, eta] = clusXTl[phi-1, eta] + clusXTl[phi+1, eta] + clusXTl[phi, eta+1] + clusXTl[phi-1, eta+1] + clusXTl[phi+1, eta+1]
                XTr_Cluster[phi, eta] = clusXTr[phi-1, eta] + clusXTr[phi+1, eta] + clusXTr[phi, eta+1] + clusXTr[phi-1, eta+1] + clusXTr[phi+1, eta+1]

                # last column
            if eta == j-1 and 0 < phi <  i-1:        
                XTc_Cluster[phi, eta] = clusXTc[phi-1, eta] + clusXTc[phi+1, eta] + clusXTc[phi, eta-1]
                XTl_Cluster[phi, eta] = clusXTl[phi-1, eta] + clusXTl[phi+1, eta] + clusXTl[phi, eta-1] + clusXTl[phi-1, eta-1] + clusXTl[phi+1, eta-1]
                XTr_Cluster[phi, eta] = clusXTr[phi-1, eta] + clusXTr[phi+1, eta] + clusXTr[phi, eta-1] + clusXTr[phi-1, eta-1] + clusXTr[phi+1, eta-1]

                # corner 00
            if phi == 0 and eta == 0:
                XTc_Cluster[phi, eta] = clusXTc[phi+1, eta] + clusXTc[phi, eta+1]
                XTl_Cluster[phi, eta] = clusXTl[phi+1, eta] + clusXTl[phi, eta+1] + clusXTl[phi+1, eta+1]
                XTr_Cluster[phi, eta] = clusXTr[phi+1, eta] + clusXTr[phi, eta+1] + clusXTr[phi+1, eta+1]

                # corner 0j
            if eta == j-1 and phi == 0:
                XTc_Cluster[phi, eta] = clusXTc[phi, eta-1] + clusXTc[phi+1, eta]
                XTl_Cluster[phi, eta] = clusXTl[phi, eta-1] + clusXTl[phi+1, eta] + clusXTl[phi+1, eta-1]
                XTr_Cluster[phi, eta] = clusXTr[phi, eta-1] + clusXTr[phi+1, eta] + clusXTr[phi+1, eta-1]

                # corner i0
            if phi == i-1 and eta == 0:
                XTc_Cluster[phi, eta] = clusXTc[phi-1, eta] + clusXTc[phi, eta+1]
                XTl_Cluster[phi, eta] = clusXTl[phi-1, eta] + clusXTl[phi, eta+1] + clusXTl[phi-1, eta+1]
                XTr_Cluster[phi, eta] = clusXTr[phi-1, eta] + clusXTr[phi, eta+1] + clusXTr[phi-1, eta+1]

                # corner ij
            if phi == i-1 and eta == j-1:
                XTc_Cluster[phi, eta] = clusXTc[phi, eta-1] + clusXTc[phi-1, eta]
                XTl_Cluster[phi, eta] = clusXTl[phi, eta-1] + clusXTl[phi-1, eta] + clusXTl[phi-1, eta-1]
                XTr_Cluster[phi, eta] = clusXTr[phi, eta-1] + clusXTr[phi-1, eta] + clusXTr[phi-1, eta-1]

        return XTc_Cluster, XTl_Cluster, XTr_Cluster
    
        ## ----------------------
        ## ----------------------  
    def Reta(self):
        rEtaTruth, rEtaXT = [], []

            ## Reta is a paramater that evaluate the ratio: E(3x7).sum()/E(7x7).sum()
            ## First set the most restrict number of clusters (7x7)
            ## Set the clusters size to:size 7x7
            ## This variable is to keep the same set of cluters 7x7 to extract clusters 3x7

            ## Set clusters to size: 3x7        
        size_eta, size_phi = 3, 7
        outputs = self.GetCluster_mxn(size_eta=size_eta, size_phi=size_phi)           
        #self.BuildClusters( size_eta=3, size_phi=7, index=self.rEtaIdx )

        clusE_3x7      = outputs[0]
        clusXTc_3x7    = outputs[1]
        clusXTl_3x7    = outputs[2]
        clusXTr_3x7    = outputs[3]
        clusNoise_3x7  = outputs[4]

        clusE_7x7      = self.ShowerShapes.clusE_7x7
        clusXTc_7x7    = self.ShowerShapes.clusXTc_7x7
        clusXTl_7x7    = self.ShowerShapes.clusXTl_7x7
        clusXTr_7x7    = self.ShowerShapes.clusXTl_7x7
        clusNoise_7x7  = self.ShowerShapes.clusNoise_7x7
        
        for i in range(len( self.idxClus7x7 )):    
            rEtaTruth.append( clusE_3x7[i].sum()/clusE_7x7[i].sum() )

            rEtaXT.append(( clusXTc_3x7[i].sum() + clusXTl_3x7[i].sum() + clusXTr_3x7[i].sum() + clusNoise_3x7[i].sum() + clusE_3x7[i].sum()) / ( clusXTc_7x7[i].sum() + clusXTl_7x7[i].sum() + clusXTr_7x7[i].sum() + clusNoise_7x7[i].sum() + clusE_7x7[i].sum() ))

        self.ShowerShapes.clusE_3x7      = clusE_3x7
        self.ShowerShapes.clusXTc_3x7    = clusXTc_3x7
        self.ShowerShapes.clusXTl_3x7    = clusXTl_3x7
        self.ShowerShapes.clusXTr_3x7    = clusXTr_3x7
        self.ShowerShapes.clusNoise_3x7  = clusNoise_3x7
        
            ## RetaTruth: Parameter evaluated with XT effects
        self.ShowerShapes.rEtaTruth = rEtaTruth
            ## RetaXT: Parameter evaluated with XT effects
        self.ShowerShapes.rEtaXT    = rEtaXT
            ## ---------------------------
            ## Plot Reta signals
        #self.PlotSS('LZT_clusReta', 'Reta', [f'E', f'E+XT'], rEtaTruth, rEtaXT)
        #PlotSS(self, pathOut, typeSS, fileName, label, ssTruth, ssXT):
        
        ## ----------------------
        ## ----------------------
    def Rphi(self):
        rPhiTruth, rPhiXT = [], []
            ## Rphi is a paramater that evaluate the ratio: E(3x3).sum()/E(3x7).sum()
            ## First set the most restrict number of clusters (3x7)

            ## Set clusters to size: 3x3
        size_eta, size_phi = 3, 3
        outputs = self.GetCluster_mxn(size_eta=size_eta, size_phi=size_phi)

        #self.BuildClusters( size_eta=3, size_phi=7, index=self.rEtaIdx )

        clusE_3x3      = outputs[0]
        clusXTc_3x3    = outputs[1]
        clusXTl_3x3    = outputs[2]
        clusXTr_3x3    = outputs[3]
        clusNoise_3x3  = outputs[4]

        clusE_3x7     = self.ShowerShapes.clusE_3x7
        clusXTc_3x7   = self.ShowerShapes.clusXTc_3x7
        clusXTl_3x7   = self.ShowerShapes.clusXTl_3x7
        clusXTr_3x7   = self.ShowerShapes.clusXTr_3x7
        clusNoise_3x7 = self.ShowerShapes.clusNoise_3x7

        
        for i in range(len( self.idxClus7x7 )):    
            rPhiTruth.append( clusE_3x3[i].sum()/clusE_3x7[i].sum() )

            rPhiXT.append(( clusXTc_3x3[i].sum() + clusXTl_3x3[i].sum() + clusXTr_3x3[i].sum() + clusNoise_3x3[i].sum() + clusE_3x3[i].sum() )/( clusXTc_3x7[i].sum() + clusXTl_3x7[i].sum() + clusXTr_3x7[i].sum() + clusNoise_3x7[i].sum() + clusE_3x7[i].sum() ) )  

        self.ShowerShapes.clusE_3x3      = clusE_3x3
        self.ShowerShapes.clusXTc_3x3    = clusXTc_3x3
        self.ShowerShapes.clusXTl_3x3    = clusXTl_3x3
        self.ShowerShapes.clusXTr_3x3    = clusXTr_3x3
        self.ShowerShapes.clusNoise_3x3  = clusNoise_3x3
        
        self.ShowerShapes.rPhiTrue = rPhiTruth
        self.ShowerShapes.rPhiXT   = rPhiXT
        #self.PlotSS('LZT_clusReta', 'R$_{{\eta}}', rEtaTruth, rEtaXT)
        
        ## ----------------------
        ## ----------------------
    def Weta2(self):
        Eeta2_xt, Eeta2, Eeta, Eeta_xt, E, E_xt, wEta2Truth, wEta2XT = [], [], [], [], [], [], [], []

        ## Set the clusters size to compute the Weta2: Cluster size 3x5
        size_eta, size_phi = 3, 5
        idxClusters  = self.SetClusterSize( size_eta=size_eta, size_phi=size_phi )
        self.BuildClusters( size_eta=3, size_phi=5 )
        #self.ComputeXT()     
        nCores = psutil.cpu_count(logical=False)
        with Pool(nCores) as pool:
            res = pool.map(self.ComputeXTmultiCore, range(len(idxClusters)) )        
            pool.close()
            pool.join()

        clusE_3x5      = self.clusEtruth
        clusXTc_3x5    = [res[i][0].reshape(size_eta, size_phi) for i in range(len(res))]
        clusXTl_3x5    = [res[i][1].reshape(size_eta, size_phi) for i in range(len(res))]
        clusXTr_3x5    = [res[i][2].reshape(size_eta, size_phi) for i in range(len(res))]
        clusNoise_3x5  = [res[i][3].reshape(size_eta, size_phi) for i in range(len(res))]

        self.ShowerShapes.Eta_3x5    = self.setEtaTruth
        self.ShowerShapes.Phi_3x5    = self.setPhiTruth
        Eta_3x5 = self.ShowerShapes.Eta_3x5
        for i in range(len(idxClusters)):
            ## Partial steps to calculate the Weta2 parameter:
            ## sqrt( (E*eta^2).sum()/E.sum() - ((E*eta).sum()/E.sum())^2)
            if len( Eta_3x5[i] ) == 3:
                Eeta2.append( ((clusE_3x5[i]+clusNoise_3x5[i])*array(Eta_3x5[i])[:,None]**2).sum() )
                Eeta.append( ((clusE_3x5[i]+clusNoise_3x5[i])*array(Eta_3x5[i])[:,None]).sum() )
                E.append((clusE_3x5[i]+clusNoise_3x5[i]).sum())

                Eeta2_xt.append( ((clusE_3x5[i] + clusNoise_3x5[i] + clusXTc_3x5[i] + clusXTl_3x5[i] + clusXTr_3x5[i]) * array(Eta_3x5[i])[:,None]**2).sum() )

                Eeta_xt.append( ((clusE_3x5[i] + clusNoise_3x5[i] + clusXTc_3x5[i] + clusXTl_3x5[i] + clusXTr_3x5[i]) * array(Eta_3x5[i])[:,None]).sum() )

                E_xt.append( (clusE_3x5[i]+clusXTc_3x5[i]+clusXTl_3x5[i]+clusXTr_3x5[i]+clusNoise_3x5[i] ).sum())
            else :
                pass
        
        self.ShowerShapes.clusE_3x5      = clusE_3x5
        self.ShowerShapes.clusXTc_3x5    = clusXTc_3x5
        self.ShowerShapes.clusXTl_3x5    = clusXTl_3x5
        self.ShowerShapes.clusXTr_3x5    = clusXTr_3x5
        self.ShowerShapes.clusNoise_3x5  = clusNoise_3x5
        
        self.ShowerShapes.wEta2Truth = [w for w in (np.sqrt( array(Eeta2)/E - (array(Eeta)/E)**2 )) if ~np.isnan(w)]
        self.ShowerShapes.wEta2XT    = [w for w in (np.sqrt( array(Eeta2_xt)/E_xt - (array(Eeta_xt)/E_xt)**2 )) if ~np.isnan(w)]

        #self.PlotSS('LZT_clusReta', 'W$_{{\eta 2}}$', wEta2Truth, wEta2XT)
        
        ##################################          
    class ShowerShapes:
        def __init__(self):
            self.rEtaTruth   = []
            self.rEtaXT      = []
            self.rPhiTruth   = []
            self.rPhiXT      = []
            self.rWeta2Truth = []
            self.rWeta2XT    = []
            
    class XTsignals:
        def __init__(self):
            self.XTc   = []
            self.Noise = []
            self.XTl   = []
            self.G     = []
            self.DeriG = []

##===========================================================
### Autocorrelation function
def autocorr(x):
    x = np.array(x)
    nRow = x.shape[0]    
    AutoCorrMatrix = np.zeros([nRow, nRow])
    try:
        nCol = x.shape[1]
    except IndexError:
        nCol = 1
    for k in range(0,nRow):    
        for t in range(0,nRow-k):
            AutoCorrMatrix[0][k] += (x-np.mean(x))[t]*(x-np.mean(x))[t+k]/sum((x-np.mean(x))**2)                    
    #Creating Autocorrelation matrix from vector of correlation lags
    for h in range(1,nRow):
        for i in range(1,nRow):
            for j in range(nRow):
                if ( i == j ):
                    AutoCorrMatrix[i][j] = AutoCorrMatrix[0][0]
                elif (abs(i-j) == h):
                    AutoCorrMatrix[i][j] = AutoCorrMatrix[0][h]  
        
    return AutoCorrMatrix

##===========================================================
def cellFunction(x):
    """
    Par(0, g_taud);
    Par(1, g_taupa);
    Par(2, g_td);
    Par(3, g_Rf);
    Par(4, g_C1);
    """    
    g_taud    =   15.82
    g_taupa   =   17.31
    g_td      =  420.00
    g_Rf      =    0.078
    g_C1      =   50.00
    par = np.zeros(5)
    par[0], par[1], par[2], par[3], par[4] = g_taud, g_taupa, g_td, g_Rf, g_C1    

    return ((exp(-x/par[0])*x*x)/(2 *par[0]*par[0]*(par[0] - par[1])) - (exp(-(x/par[0]))*x*par[1])/(par[0]*pow(par[0] - par[1],2)) + exp(-(x/par[0]))*par[1]*par[1]/pow(par[0] - par[1],3) + (exp(-(x/par[1]))*par[1]*par[1])/pow(-par[0] + par[1],3) + (1/(2*par[2]*par[0] *pow((par[0] - par[1]),3)))*exp(-x* (1/par[0] + 1/par[1]))* (-2 *exp(x *(1/par[0] + 1/par[1]))*par[0] *pow((par[0] - par[1]),3) - 2 *exp(x/par[0])*par[0]*pow(par[1],3) + exp(x/par[1]) *(x*x *pow((par[0] - par[1]),2) + 2*x*par[0]*(par[0]*par[0] - 3*par[0]*par[1] + 2*par[1]*par[1]) + 2*par[0]*par[0]*(par[0]*par[0] - 3*par[0]*par[1] + 3*par[1]*par[1]))) + ((1 - (exp((-x + par[2])/par[0])*(x - par[2])*(par[0] - 2*par[1]))/pow((par[0] - par[1]),2) - (exp((-x + par[2])/par[0])*(x - par[2])*(x- par[2]))/(2*par[0]*(par[0] - par[1])) + (exp((-x + par[2])/par[1])*par[1]*par[1]*par[1])/pow((par[0] - par[1]),3) - (exp((-x + par[2])/par[0])*par[0]*(par[0]*par[0] - 3*par[0]*par[1] + 3*par[1]*par[1]))/pow((par[0] - par[1]),3))* heaviside(x -par[2],1))/par[2])*par[3]*par[4]*par[3]*par[0]*par[0]
    
## ==========================================================
def createPath(pathName):
    if not os.path.exists(pathName):
        #os.mkdir(f'{pathName}')
        pathlib.Path(pathName).mkdir(parents=True, exist_ok=True) 

    return pathName


##===========================================================
class DateInfo:
    def __init__(self):
        self.GMT = -3    ## Brazilian Time zone
        self.getTime()

    def getTime(self):
        now = datetime.now().astimezone(timezone(timedelta(hours=self.GMT)))
        self.day      = now.strftime('%d')
        self.strDay   = now.strftime('%a')
        self.month    = now.strftime('%m')
        self.strMonth = now.strftime('%b')
        self.year     = now.strftime('%Y')
        self.date     = now.strftime('%Y%m%d')
        self.dates    = now.strftime('%Y%b%d')
        self.hour     = now.strftime('%Hh%Mm%Ss')
        self.dateHour = now.strftime('%Y%m%d_%Hh%Mm%Ss')   
        self.fullTime = now.strftime('%Hh %Mm %Ss %Y/%m/%d') 


##===========================================================
def derivCellFunction(x):
    return derivative(cellFunction, x, dx=1e-6)

##===========================================================
def derivXtalkFunction(x):
    return derivative(derivXtalkFunction, x, dx=1e-6)

##===========================================================
def loadSaveDict(filename, **kwargs):
    dataDict = kwargs.get('dataDict')
    load = kwargs.get('load')
    save = kwargs.get('save')
    
    if load is not None:
        if os.path.exists(filename):  # Check if the file exists
            with open(filename, "rb") as file:
                loadedDict = pkl.load(file)
            if not loadedDict:
                print("The file is empty.")
            else:
                return loadedDict
        else:
            print("The file does not exist.")
            return 0  # Handle the case where the file doesn't exist
        
    if save is not None:
        with open(filename, "wb") as file:
            pickle.dump(dataDict, file)   


## =======================================
def getIdxClus_mxn(cluster, m, n):
    #m, n = 5,5
    row, col = cluster.shape[0] // 2, cluster.shape[1] // 2    
    idx_mxn = cluster[row - m // 2:row + m // 2 + 1, col - n // 2:col + n // 2 + 1]
    
    return idx_mxn.flatten()

## =======================================
def getIdxSampClus(nCells, n):
    """
        Return the index for a cluster mx m*nSamp given the new order n
    """
    nSamp = 4
    
    m = int(np.sqrt( nCells/nSamp ) )

    array_2d = array( range(nCells) ).reshape( m, m*nSamp )

    # Calculate the center indices
    c_row = array_2d.shape[0] // 2
    c_col = array_2d.shape[1] // 2

    # Extract the central values
    idx = array_2d[
        c_row - n//2 : c_row + n-n//2,
        c_col - (n*nSamp)//2 : c_col + (n*nSamp)//2
    ]   
    
    return list(idx.flatten())    

##########################################
def getSize(filename):
    try :
        file_size = os.path.getsize(filename)
        mega  = int(file_size/2**(20))
        giga  = int(file_size/2**(30))
        kilo  = int(file_size/2**(10))
        if giga == 0:
            if mega == 0:
                if kilo == 0:
                    return f'{file_size:.2f} Bytes'
                else :
                    return f'{file_size/2**(10):.2f} KB'
            else :    
                return f'{file_size/2**(20):.2f} MB'
        else :
            return f'{file_size/2**(30):.2f} GB'
                
    except :
        return f'{filename} not found!'

## ==========================================================
def genRandVal(count, start, end):
    values = set()
    while len(values) < count:
        value = random.randint(start, end)
        values.add(value)
    return list(sorted(values, reverse=True))
    
## ==========================================================
def loadEbinData(source, **kwargs):
    energy  = kwargs.get('energy')    
    label   = kwargs.get('label')
    clus3x3 = kwargs.get('clus3x3')
    debug   = kwargs.get('debug')
    
    if clus3x3 is None: clus3x3 = 'no'
    if debug is None: debug = 'no'

    if clus3x3.lower() == 'yes': clusIdx = [i for i in range(24, 36)] + [i for i in range(44, 56)] + [i for i in range(64, 76)]
    else: clusIdx = [i for i in range(100)]

    sampSignals = np.concatenate(([np.load( f, allow_pickle=True ) for f in glob.glob( f'{source}/{energy}/*Samples*.npy')]), axis=1)
    signals     = np.concatenate(([np.load( f, allow_pickle=True ) for f in glob.glob(f'{source}/{energy}/*Amplitudes*.npy')]), axis=1)    

    EtruthSamp, XTcSamp, XTlSamp, XTrSamp, NoiseSamp = sampSignals[0], sampSignals[1], sampSignals[2], sampSignals[3], sampSignals[4]
    Etruth   = array( [e for e in signals[0]])
    if debug.lower() == 'yes':
        nSamples = 330
    else :
        nSamples = len( Etruth )
    
    #nSamples = 330

    xData    = np.add( np.add(np.add( EtruthSamp[:nSamples, clusIdx], XTcSamp[:nSamples, clusIdx]), XTlSamp[:nSamples, clusIdx]), NoiseSamp[:nSamples, clusIdx] )

    AmpTime   = OptFilt( EtruthSamp )
    yData     = EtruthSamp[:nSamples, clusIdx]
    AmpTimeXT = OptFilt( xData )

    #OFdictXT   = {'AmpClusters':AmpTimeXT['Clusters']['Amplitude'], 'TimeClusters':AmpTimeXT['Clusters']['Time'], 'RawData':AmpTimeXT['Clusters']['RawData'] }
    #OFdictTrue = {'AmpClusters':AmpTime['Clusters']['Amplitude'], 'TimeClusters':AmpTime['Clusters']['Time'], 'RawData':AmpTime['Clusters']['RawData'] }

    OFdictXT   = {'AmpClusters':AmpTimeXT['Clusters']['Amplitude'], 'TimeClusters':AmpTimeXT['Clusters']['Time']}
    OFdictTrue = {'AmpClusters':AmpTime['Clusters']['Amplitude'], 'TimeClusters':AmpTime['Clusters']['Time']}

    return xData, yData, EtruthSamp[:nSamples, clusIdx], nSamples, OFdictXT, OFdictTrue 

## ==========================================================
def loadEbinDict(pathSamp, **kwargs):   
    energy  = kwargs.get('energy')
    data    = kwargs.get('data')
    pathAmp = kwargs.get('pathAmp') 
    clus3x3 = kwargs.get('clus3x3')
    debug   = kwargs.get('debug')
    train_type = kwargs.get('train_type')

    if data.lower() == 'lzt':
        sampFile = f'{pathSamp}/{energy}/*11k*{energy}.pkl'
        sampAmp  = f'{pathAmp}/{energy}/*{energy}.pkl'

        signSamples = loadSaveDict( glob.glob( sampFile)[0] ,load=True )
        signAmplitude = loadSaveDict( glob.glob( sampAmp )[0] ,load=True )
        EtruthAmp, _, _, _, _, _, _, _  = signAmplitude.values()

        EtruthSamp, XTcSamp, XTlSamp, _, NoiseSamp, _, _ = signSamples.values()

    else:
        sampFile = f'{pathSamp}/{energy}/*Samp*11k.pkl'
        sampAmp  = f'{pathAmp}/{energy}/*Amplit*11k.pkl'

        signSamples = loadSaveDict( glob.glob( sampFile)[0] ,load=True )
        signAmplitude = loadSaveDict( glob.glob( sampAmp )[0] ,load=True )

        EtruthSamp, _, XTcSamp, XTlSamp, _, NoiseSamp, _, _, _, _ = signSamples.values()
        EtruthAmp, _, _, _, _, _, _,  = signAmplitude.values()

    if train_type is None: train_type = 'regression'    
## Select between filter mode to regress samples information or regression mode to regress amplitude or time of the cells        
    
    if debug.lower() == 'yes':
        nSamples = 330
    else :
        nSamples = len( EtruthSamp )

    NoiseSamp = []
    for _ in range( nSamples ):
        NoiseSamp.append( genNoise(EtruthSamp.shape[1]) )

    NoiseSamp = array( NoiseSamp )

    print(train_type)

    if train_type.lower() == 'filter':

        if clus3x3.lower() == 'yes': clusIdx = getIdxSampClus(EtruthSamp.shape[1], 3)
        else : clusIdx = getIdxSampClus(EtruthSamp.shape[1], 5)

        yData     = EtruthSamp[:nSamples, clusIdx]

    elif train_type.lower() == 'regression':
        print('  >>>> inside if train_type')
        clusIdx = getIdxSampClus(EtruthSamp.shape[1], 5)
                
        idx_7x7 = array(range(49))
        idx_5x5 = getIdxClus_mxn(idx_7x7.reshape(7,7), 5, 5)
        idx_3x3 = getIdxClus_mxn(idx_7x7.reshape(7,7), 3, 3)
        
        if data.lower() == 'lzt':
            yData     = EtruthAmp[:nSamples, idx_5x5]/1e3
        if data.lower() == 'emshower':
            yData     = EtruthAmp[:nSamples, :]/1e3

    else:
        print(f'You must choose regression or filter mode to train the NN! ')
        sys.exit(1)

    xData    = np.add( np.add(np.add( EtruthSamp[:nSamples, clusIdx], XTcSamp[:nSamples, clusIdx]), XTlSamp[:nSamples, clusIdx]), NoiseSamp[:nSamples, clusIdx] )

    AmpTime   = OptFilt( EtruthSamp )
    AmpTimeXT = OptFilt( xData )

    OFdictXT   = {'AmpClusters':AmpTimeXT['Clusters']['Amplitude'], 'TimeClusters':AmpTimeXT['Clusters']['Time']}
    OFdictTrue = {'AmpClusters':AmpTime['Clusters']['Amplitude'], 'TimeClusters':AmpTime['Clusters']['Time']}

    return xData, yData, EtruthSamp[:nSamples, clusIdx], nSamples, OFdictXT, OFdictTrue 
##===========================================================
def loadData(data, source, **kwargs):
    e_range = kwargs.get('e_bins')
    energy  = kwargs.get('energy')    
    label   = kwargs.get('label')
    clus3x3 = kwargs.get('clus3x3')
    debug   = kwargs.get('debug')

    if clus3x3 is None: clus3x3 = 'no'

    if clus3x3.lower() == 'yes': 
        clusIdx  = [i for i in range(24, 36)] + [i for i in range(44, 56)] + [i for i in range(64, 76)]        
        cellsIdx = [6, 7, 8, 11, 12, 13, 16, 17, 18]
    else:     
        clusIdx  = [i for i in range(100)]
        cellsIdx = [i for i in range(25)]
        
    if label is None: label=''    
    print('{:^30}'.format("*"*30))
    
    #### -----------------------;
    ##   EMSHOWER
    #### -----------------------;
    if data.lower() == 'emshower':        
        print('Loading EMshower data...')        
        fileEtruth     = f'{source}/Clusters_Etruth_745k.txt'
        fileEtruthSamp = f'{source}/Clusters_EtruthSamp_745k.txt'
        fileXTclsamp   = f'{source}/Clusters_XTclSamp_745k.txt'
        fileNoisesamp  = f'{source}/Clusters_NoiseSamples_745k.txt'
        Etruth       = array(pd.read_csv(fileEtruth,     delim_whitespace=True, header=None))
        EtruthSamp   = array(pd.read_csv(fileEtruthSamp, delim_whitespace=True, header=None))
        XTclSamp     = array(pd.read_csv(fileXTclsamp,   delim_whitespace=True, header=None))
        NoiseSamp    = array(pd.read_csv(fileNoisesamp,  delim_whitespace=True, header=None))
        nSamples     = 440000
        energy       = None
        detail       = ''
        #xData        = EtruthSamp[:nSamples, :]+XTclSamp[:nSamples, :]+NoiseSamp[nSamples, :]
        xData        = EtruthSamp[:nSamples, clusIdx]+XTclSamp[:nSamples, clusIdx]+NoiseSamp[nSamples, clusIdx]
        XTclNoise    = np.add( XTclSamp[:nSamples, clusIdx], NoiseSamp[:nSamples, clusIdx] )

    #### -----------------------;
    ##   LZT
    #### -----------------------;
    elif data.lower() == 'lzt':
        print('Loading LZT data...')
        if debug.lower() == 'yes':
            nClusPerFile = 4*11
        else :
            nClusPerFile = 4*11*750        
        all_binsSamples, all_binsSignals = [], []
        for energy in e_range:
        # Load and concatenate the data from the .npy files into a single numpy array
            #concatenated_data = np.concatenate([np.load(os.path.join(folder_path, file)) for file in file_list], axis=1)
            binSamples = np.concatenate(([np.load( f, allow_pickle=True ) for f in glob.glob( f'{source}/{energy}/*Samples*.npy')]), axis=1)
            binSignals = np.concatenate(([np.load( f, allow_pickle=True ) for f in glob.glob( f'{source}/{energy}/*Amplitudes_*.npy')]), axis=1)

            # Shuffle the data along the second axis (axis=1) to randomly mix events from different files
            np.random.shuffle(binSamples.T)
            np.random.shuffle(binSignals.T)

            # Choose the first nClusPerFile events from the shuffled data
            nBinSamples = binSamples[:, :nClusPerFile]
            nBinSignals = binSignals[:, :nClusPerFile]

            # Append the random nBins to the list
            all_binsSamples.append(nBinSamples)
            all_binsSignals.append(nBinSignals)

        # Concatenate all the random_10k_events arrays obtained during the loop
        signals     = np.concatenate(all_binsSignals, axis=1)
        sampSignals = np.concatenate(all_binsSamples, axis=1)

        #sampSignals = np.concatenate(([np.load( glob.glob( f'{source}/{energy}/*EXTc*Samples_*.npy')[0],    allow_pickle=True ) for energy in e_range ]), axis=1)
        #signals     = np.concatenate(([np.load( glob.glob( f'{source}/{energy}/*EXTc*Amplitudes_*.npy')[0], allow_pickle=True ) for energy in e_range ]), axis=1)
        
        EtruthSamp, XTcSamp, XTlSamp, XTrSamp, NoiseSamp = sampSignals[0], sampSignals[1], sampSignals[2], sampSignals[3], sampSignals[4]

        Etruth    = array([e for e in signals[0]])
        nSamples  = len(Etruth)        
        #xData     = np.add(np.add(np.add( EtruthSamp[:nSamples,:], XTcSamp[:nSamples,:]), XTlSamp[:nSamples,:]), NoiseSamp[:nSamples,:] )
        #XTclNoise = np.add(np.add(XTcSamp[:nSamples,:], XTlSamp[:nSamples,:]), NoiseSamp[:nSamples,:] )

        xData     = np.add(np.add(np.add( EtruthSamp[:, clusIdx], XTcSamp[:, clusIdx]), XTlSamp[:, clusIdx]), NoiseSamp[:, clusIdx] )
        XTclNoise = np.add(np.add(XTcSamp[:, clusIdx], XTlSamp[:, clusIdx]), NoiseSamp[:, clusIdx] )

        detail    = f'_{label}'
    else:
        print(f'-d or --data should be defined to load emshower, lzt or lztEbins dataset! ')
        sys.exit(1)    
    print('{:^30}'.format("*"*30))        
    AmpTime   = OptFilt( EtruthSamp[:nSamples, clusIdx] )
    InputDim  = np.shape(xData)[1]
    
    return xData, energy, Etruth[:, cellsIdx], EtruthSamp[:, clusIdx], clusIdx, nSamples, AmpTime, InputDim, detail, XTclNoise

##===========================================================
def loadSignals_old(filesSamples, filesAmplitudes):

    sampSignals = np.load( f'{filesSamples[0]}', allow_pickle='TRUE')    
    sampSignals = np.concatenate((sampSignals, np.load( f'{filesSamples[1]}', allow_pickle='TRUE')), axis=1)
    sampSignals = np.concatenate((sampSignals, np.load( f'{filesSamples[2]}', allow_pickle='TRUE')), axis=1)
    sampSignals = np.concatenate((sampSignals, np.load( f'{filesSamples[3]}', allow_pickle='TRUE')), axis=1)
    
    #sampSignals = [np.concatenate((sampSignals, np.load( f'{filesSamples[i]}', allow_pickle='TRUE')), axis=1) for i in (1,filesSamples.shape(1))]

    signals = np.load( f'{filesAmplitudes[0]}', allow_pickle='TRUE')
    signals = np.concatenate((signals, np.load( f'{filesAmplitudes[1]}', allow_pickle='TRUE')), axis=1)
    signals = np.concatenate((signals, np.load( f'{filesAmplitudes[2]}', allow_pickle='TRUE')), axis=1)
    signals = np.concatenate((signals, np.load( f'{filesAmplitudes[3]}', allow_pickle='TRUE')), axis=1)

    return

##===========================================================
def loadSignals(source, e_range):
    
    sampSignals = np.concatenate(array([np.load( f, allow_pickle='TRUE') for f in glob.glob(f'{source}/{e_range}/*EXTc*Samples_*.npy')]), axis=1)
    signals     = np.concatenate(array([np.load( f, allow_pickle='TRUE') for f in glob.glob(f'{source}/{e_range}/*EXTc*Amplitudes_*.npy')]), axis=1)    

    return sampSignals, signals
##===========================================================
## Optinal Filtering (Baseline)
def OptFilt_old(samples, ai, bi):
    signals = int(samples.shape[1]/4)
    AmpTime = dict()
    AmpRec, TimeRec = np.zeros([samples.shape[0],signals]), np.zeros([samples.shape[0],signals])
    for cell in range(signals):
        AmpRec[:, cell]  = samples[:, 0+4*cell: 4+4*cell]@ai
        TimeRec[:, cell] = samples[:, 0+4*cell: 4+4*cell]@bi/AmpRec[:, cell]
        AmpTime.update({f'Cell {cell+1}': {'Amplitude':AmpRec[:,cell],'Time':TimeRec[:,cell] }})        
    return AmpTime

##===========================================================
## Optinal Filtering (Baseline)
def OptFilt(samples, ai=None, bi=None):
    idx_7x7 = array(range(49))
    idx_5x5 = getIdxClus_mxn(idx_7x7.reshape(7,7), 5, 5)
    idx_3x3 = getIdxClus_mxn(idx_7x7.reshape(7,7), 3, 3)
    ij_cell = ['-3,3' , '-2,3' , '-1,3' , '0,3' , '1,3' , '2,3' , '3,3' , 
           '-3,2' , '-2,2' , '-1,2' , '0,2' , '1,2' , '2,2' , '3,2' , 
           '-3,1' , '-2,1' , '-1,1' , '0,1' , '1,1' , '2,1' , '3,1' , 
           '-3,0' , '-2,0' , '-1,0' , '0,0' , '1,0' , '2,0' , '3,0' , 
           '-3,-1', '-2,-1', '-1,-1', '0,-1', '1,-1', '2,-1', '3,-1', 
           '-3,-2', '-2,-2', '-1,-2', '0,-2', '1,-2', '2,-2', '3,-2', 
           '-3,-3', '-2,-3', '-1,-3', '0,-3', '1,-3', '2,-3', '3,-3' ]
    if samples.shape[1] == 25 : ij_cell = ij_cell[idx_5x5]
    if samples.shape[1] == 9 : ij_cell = ij_cell[idx_3x3]
    if ai is None:        
        ##produces a tau_0 = -0.89
        #ai = [  0.36308142, 0.47002328, 0.39304565,  0.30191008]
        #bi = [-20.77449385, 5.48756441, 6.21710107, 10.33539619]

        ## Mykola coeffs
        ## produces a tau_0 = 0.04 on LZT data
        #ai = [0.3594009, 0.49297974, 0.38133506, 0.24622458]
        #bi = [-18.92073871, 0.90162148, 14.33011022, 6.34564695]

        ## the same result for Mikola coeffs
        ai = [0.3227028, 0.5390962, 0.3881769, 0.1247128]
        bi = [-15.31452, -4.724544, 16.791813, 14.272150]
        
    nSamp = len(ai)
    signals = int(samples.shape[1]/nSamp)
    AmpTime = dict()          
    
    AmpRec  = np.tensordot(samples.reshape(samples.shape[0], signals, nSamp),ai, axes=(2,0))    
    TimeRec = (np.tensordot(samples.reshape(samples.shape[0], signals, nSamp), bi, axes=(2,0)))/AmpRec

    AmpTime = {'Cells': { f'Cell {ij_cell[i]}':{'Amplitude':AmpRec[:,i], 'StdAmp':np.std(AmpRec[:,i]), 'Time': TimeRec[:,i], 'StdTime':np.std(TimeRec[:,i])} for i in range(signals)}}
    
    AmpTime.update({'Clusters': {'Std': {'Amp': np.sum(AmpRec, axis=0).std(), 'Time':TimeRec.mean(axis=0).std()}, 
                                 'Mean':{ 'Time':TimeRec.mean(axis=0),'Amp': AmpRec.mean(axis=0)},
                                 'SumAmplitudes':AmpRec.sum(axis=1), 'Amplitude':AmpRec,'Time':TimeRec, 'RawData': samples}})
    return AmpTime

##===========================================================
def OF_coeffs(inNoiseMatrix, inSi, inGsig, inDgSig):
    AmpTime, OF, Verif = dict(), dict(), dict()
   
    Si    = inSi
    gSig  = inGsig
    DgSig = inDgSig
    NoiseMatrix = inNoiseMatrix

    #AmpTime = dict();    
    nRow = inNoiseMatrix.shape[0]    
    Rxx  = np.zeros([nRow, nRow])            

    Rxx = autocorr(NoiseMatrix)
    Rxx = np.linalg.inv(Rxx)
    Q1  = ((gSig.T@Rxx)@gSig)
    Q2  = ((DgSig.T@Rxx)@DgSig)
    Q3  = ((DgSig.T@Rxx)@gSig)

    Delta = Q1 * Q2 - Q3**2
    mu    = Q3/Delta
    lambd = Q2/Delta
    ro    = -Q1/Delta
    k     = -Q3/Delta

    aCoef = lambd*(Rxx@gSig) + k*(Rxx@DgSig)
    bCoef = mu*(Rxx@gSig) + ro*(Rxx@DgSig)             

    Amp = (aCoef.T@Si)/1000 ## /1e3 => GeV
    t   = (bCoef.T@Si)/(aCoef.T@Si)                        

    # Verification of Optimal Algorithm 
    MustBe1 = lambd*((gSig.T@Rxx)@gSig) + k*((gSig.T@Rxx)@DgSig)
    MustBe0 = lambd*((DgSig.T@Rxx)@gSig) + k*((DgSig.T@Rxx)@DgSig)
    
    print(f'Must to be 1: {MustBe1} | Must to be 0: {MustBe0}')
    return array(aCoef), array(bCoef)

##===========================================================
## Optinal Filtering (Baseline)
def OptimalFilter(inNoiseMatrix, inSi, inGsig, inDgSig):
    """
        inNoiseMatrix - noise samples
        inSi    - cell energy samples
        inGsig  - cell signal samples
        inDgSig - first derivative signal samples
    """
    samples, cells = inSi.shape
    cells = cells/4
    print(f'=========================================')
    print(f' Energy and time reconstruction process!')
    print(f'  Samples: {samples}')
    print(f'  Cells  : {cells}')
    print(f'  Processing...')
    ## Variables are the shape: 4000 x 36. 4000 samples; 9 cells with 4 samples each one        

    ## ===============================================
    ## To store and organize data on dict
    clusterAmp = np.zeros([int(samples), int(cells)])
    clusterTime = np.zeros([int(samples), int(cells)])
    MustBe0 = np.zeros([int(samples), int(cells)])
    MustBe1 = np.zeros([int(samples), int(cells)])
    a_i = np.zeros([4, int(samples), int(cells)])
    b_i = np.zeros([4, int(samples), int(cells)])
    ## ===============================================

    ## =====================
    ## Previous version
    #ListData = []
    ## =====================
    for row in range(int(samples)):        
        #input()
        #print("Sample: ", row+1)
        AmpTime, OF, Verif = dict(), dict(), dict()
        for col in range(int(cells)):
            Q1, Q2, Q3 = 0, 0, 0
            Delta, mu, k, ro, lambd = 0, 0, 0, 0, 0
            aCoef = 0
            bCoef = 0
            #print("Cell: ", col+1)
            Si    = inSi[row][4*col+0:4*col+4]
            gSig  = inGsig[row][4*col+0:4*col+4]
            DgSig = inDgSig[row][4*col+0:4*col+4]
            NoiseMatrix = inNoiseMatrix[row][4*col+0:4*col+4]
            
            #AmpTime = dict();    
            nRow = inNoiseMatrix.shape[0]    
            Rxx  = np.zeros([nRow, nRow])            

            Rxx = autocorr(NoiseMatrix)
            Rxx = np.linalg.inv(Rxx)
            Q1  = ((gSig.T@Rxx)@gSig)
            Q2  = ((DgSig.T@Rxx)@DgSig)
            Q3  = ((DgSig.T@Rxx)@gSig)

            Delta = Q1 * Q2 - Q3**2
            mu    = Q3/Delta
            lambd = Q2/Delta
            ro    = -Q1/Delta
            k     = -Q3/Delta

            aCoef = lambd*(Rxx@gSig) + k*(Rxx@DgSig)
            bCoef = mu*(Rxx@gSig) + ro*(Rxx@DgSig)             
            
            Amp = (aCoef.T@Si)/1e3 ## /1e3 => GeV
            t   = (bCoef.T@Si)/(aCoef.T@Si)                        
        
            ## ================================
            ## Previous version without nested dict
            #AmpTime.update({f'E_Cell_{col+1}': (aCoef.T@Si), f't_Cell_{col+1}': (bCoef.T@Si)/(aCoef.T@Si)})
            ## ================================
            
            # Verification of Optimal Algorithm 
            MustBe1[row, col] = lambd*((gSig.T@Rxx)@gSig) + k*((gSig.T@Rxx)@DgSig)
            MustBe0[row ,col] = lambd*((DgSig.T@Rxx)@gSig) + k*((DgSig.T@Rxx)@DgSig)
            ## =============================
            ## This is to store and organize on dict
            a_i[:, row, col] = aCoef
            b_i[:, row, col] = bCoef

            clusterAmp[row, col] = Amp
            clusterTime[row, col] = t
            ## =============================

        ## ================================
        ## Previous version without nested dict
        #ListData.append(AmpTime)
        ## ================================
    for i in range(int(cells)):
        Verif.update({'MustBe0':MustBe0[:,i],'MustBe1': MustBe1[:,i]})
        OF.update({'a_i': a_i[:,:,i], 'b_i': b_i[:,:,i], 'Verification': Verif})
        AmpTime.update({f'Cell {i+1}': {'Energy': clusterAmp[:, i], 'Time': clusterTime[:, i], 'OptimalFilter': OF }}) 
    print(f' Reconstruction process is done!')        
    print(f' {samples} samples reconstructed!')
    print(f'    -------------------------------    \n')
    ## ================================
    ## Previous version without nested dict
    #return ListData
    ## ================================
    return AmpTime
##===========================================================
def NNmodel(InputDim, OutputDim, **kwargs):
    lossFunc = kwargs.get('lossFunc')
    nLayers  = kwargs.get('nLayers')   
    nNeurons = kwargs.get('nNeurons') 
    regul    = kwargs.get('regul')
    summary  = kwargs.get('summary')
    
    if lossFunc is None: lossFunc = 'mse'    ### for trainings before jun11
    #if lossFunc is None: lossFunc = rmse()
    if nLayers  is None: nLayers  = 1
    if summary  is None: summary  = True

    model = Sequential()
    
    #model.add(Dense(InputDim, input_dim=InputDim, kernel_initializer='normal', activation='relu'))
    model.add(Dense(InputDim, input_dim=InputDim, kernel_initializer='normal', activation=layers.LeakyReLU(alpha=0.3)))
    #model.add(Dense(InputDim, input_dim=InputDim, activation='sigmoid'))
    
    #nNeurons = np.linspace(nNeurons, int(np.sqrt(nNeurons)) + OutputDim, nLayers, dtype=int)    
    nNeurons = np.linspace(nNeurons, int(np.sqrt(nNeurons)) + int(OutputDim/2), nLayers, dtype=int)    
    comp = str()
    
    if regul is None:
        for n in nNeurons:
            model.add(Dense(n, kernel_initializer='normal', activation= layers.LeakyReLU(alpha=0.3)))
            comp = comp+f'_{n}'
    if regul:
        #print('model with regularizer!')
        for n in nNeurons:
            model.add(Dense(n, kernel_initializer='normal', 
                            activation           = layers.LeakyReLU(alpha=0.3),
                            kernel_regularizer   = regularizers.L1L2(l1=1e-3, l2=1e-3),
                            bias_regularizer     = regularizers.L2(1e-4),
                            activity_regularizer = regularizers.L2(1e-3)
                           )
                     )
            comp = comp+f'_{n}'
                
    desc = f'In{InputDim}_{nLayers}Lay{comp}_Out{OutputDim}'

    ### =======================
    ### Output Layer
    model.add(Dense(OutputDim, activation='linear'))
    model.compile(optimizer = optimizers.Adam(learning_rate = 0.001),
                  loss      = lossFunc,
                  #loss      = 'mse', 
                  #metrics   = [msle(), rmse(), 'mae', r2_score],
                  metrics   = [msle(), rmse(), 'mae', r2_score], ### for trainings before jun11
                  run_eagerly = True)
    if summary:
        model.summary()
               
    return model, desc, nNeurons


##===========================================================
def NNmodel__(InputDim, OutputDim, nLayers, nNeuron, summary=True):
    model = Sequential()
    
    #model.add(Dense(InputDim, input_dim=InputDim, activation='sigmoid'))
    model.add(Dense(InputDim, input_dim=InputDim, kernel_initializer='normal', activation=layers.LeakyReLU(alpha=0.2)))
    model.add(Dense(nNeuron, activation=layers.LeakyReLU(alpha=0.2)))
    if (nLayers == 1):
        #desc = f'In{InputDim}_{nLayers}Lay_{nNeuron}_Out{OutputDim}'
        comp = ''

    n = np.linspace()

    if (nLayers > 1):
        for n in range(1,nLayers):
            #model.add(Dense(nNeuron[n], kernel_initializer='normal', activation='relu'))
            model.add(Dense(InputDim, input_dim=InputDim, kernel_initializer='normal', activation=layers.LeakyReLU(alpha=0.3)))
    ### =======================
    ### Hidden Layers
    if (nLayers == 2 ):
        #for n in range(nLayers):
        #model.add(Dense(5, kernel_initializer='normal', activation='relu'))
        model.add(Dense(5, kernel_initializer='normal', activation=layers.LeakyReLU(alpha=0.2)))
            #model.add(Dense(nNeuron[n], activation='relu'))
        #desc = f'In{InputDim}_{nLayers}Lay_{nNeuron}_5_Out{OutputDim}'
        comp = '_5'
    if (nLayers == 3 ):
        model.add(Dense(10, kernel_initializer='normal', activation=layers.LeakyReLU(alpha=0.2)))
        model.add(Dense(5,  kernel_initializer='normal', activation=layers.LeakyReLU(alpha=0.2)))
        #desc = f'In{InputDim}_{nLayers}Lay_{nNeuron}_10_5_Out{OutputDim}'
        comp = '_10_5'
    desc = f'In{InputDim}_{nLayers}Lay_{nNeuron}{comp}_Out{OutputDim}'
    ### =======================
    ### Outup Layer
    #model.add(Dense(OutputDim, activation='linear'))
    model.add(Dense(OutputDim,  kernel_initializer='normal', activation=layers.LeakyReLU(alpha=0.2)))

    model.compile(optimizer =optimizers.Adam(learning_rate = 0.005),
                  loss      ='mse', 
                  #metrics   = ['mean_absolute_percentage_error'])    
                  metrics   = ['mae', 'mape', r2score])
    if summary:
        model.summary()
               
    return model, desc

##===========================================================
def plotBar(data, barType, **kwargs):     
    
    show      = kwargs.get('show')
    barW      = kwargs.get('barW')
    ext       = kwargs.get('ext')
    ylog      = kwargs.get('ylog')
    save      = kwargs.get('save')    
    titleName = kwargs.get('titleName')    
    fileName  = kwargs.get('fileName')    
    xlabel    = kwargs.get('xlabel')
    ylabel    = kwargs.get('ylabel')
    xLabels   = kwargs.get('xLabels')      
    
    if show is None: show = False
    if barW is None: barW = 0.45
    if ylog is None: ylog = False
    if save is None: save = False
    if ext  is None: ext  = 'pdf'        
    if xlabel  is None: xlabel = ' '
    if ylabel  is None: ylabel = ' '
    if xLabels is None: xLabels = list( range( len(data) ) )
    if titleName is None: titleName = ''
    if fileName is None: fileName = ''        
    
    from matplotlib import ticker
    Ndpi = 600
    x = np.arange(len(data))
    if (len(data) > 9):        
        x_pos = np.linspace(2,34,9)
    else :
        x_pos = np.arange(1,len(xLabels)+1)
    fig = plt.figure()
    plt.bar(x+barW/2, data, barW, label=f'{barType}')
    plt.title(f'{titleName}')
    plt.xlabel(f'{xLabel}')
    plt.ylabel(f'{yLabel}')
    plt.legend(frameon=False)  
    plt.xticks(x_pos, xLabels)
    plt.grid(linestyle='--',linewidth=.7)
    if (ylog == True):
        plt.gca().set_yscale("log")
    if ext == None:
        ext = png
    if save == True :
        fig.savefig(f'{pathOut}/{fileName}.{ext}', format=ext, dpi=Ndpi)
    if (show == False ):
        plt.close(fig)        
    elif (show == True):        
        plt.show()

##===========================================================
"""def plot_heatmap(pathOut, data, title, fileName, save=False, show=False, precis=2):
    from matplotlib import cm,colors
    fig  = plt.figure()
    n       = int(np.sqrt(len(data)))
    data    = data.reshape(n,n)
    heatmap = plt.pcolor(data, cmap=cm.YlOrBr, norm=colors.Normalize(vmin=np.amin(data)*1.5, vmax=np.amax(data)*1.5))
    xy_labels = range(1, n+1)
    #xpos    = np.linspace(1,n,n)-0.5
    #ypos    = np.linspace(1,n,n)-0.5

    for y in range(n):
        for x in range(n):
            plt.text(x + 0.5, y + 0.5, f'{data[y, x]:.{precis}f}',
                    horizontalalignment='center',
                    verticalalignment='center',
                    )

    plt.colorbar(heatmap, format=f'%.{precis}f')
    plt.xticks(array(xy_labels)-0.5,xy_labels)
    plt.yticks(array(xy_labels)-0.5,xy_labels)
    plt.title(f'{title}')

    if save == True:
        fig.savefig(f'{pathOut}/{fileName}.pdf', format='pdf')  

    if (show == False ):
        plt.close(fig)        
    elif (show == True):        
        #plt.show()
        #plt.gca().set_yscale("log")
        plt.show()"""
#def plot_heatmap(pathOut, data, title, fileName, save=False, precis=2, show=False, vMin=False, vMax=False, log=False, **kwargs):
def plot_heatmap(data, **kwargs):
    default_params = {
        "ext": "pdf",
        "fileName": "",
        "GeV": 1,
        "label": "",        
        "log": False,
        "pathOut": None,
        "precis": 2,
        "save": False,        
        "show": True,
        "titleName": "",
        "vMin": np.amin(data)*0.8,
        "vMax": np.amax(data)*1.5,
        "xy_labels": None
    }
    params = {**default_params, **kwargs}
    
    from matplotlib import cm, colors
    plt.rc('figure', titlesize=44)  # fontsize of the figure title
    #fig  = plt.figure()
    fig, ax   = plt.subplots()
    n       = int(np.sqrt(len(data)))
    data    = data.reshape(n,n)/params["GeV"]
    if params["log"] == False:
        heatmap = plt.pcolor(data, cmap=cm.YlOrBr, norm=colors.Normalize(vmin=params["vMin"], vmax=params["vMax"]))
    else :
        heatmap = plt.pcolor(data, cmap=cm.YlOrBr, norm=colors.LogNorm(vmin=params["vMin"], vmax=params["vMax"]))        
    xy_position = array(range(n))
    if params["xy_labels"] is None:
        xy_labels = xy_position
    else: 
        xy_labels = array(range(params["xy_labels"][0], params["xy_labels"][1]))

    for y in range(n):
        for x in range(n):
            plt.text(x + 0.5, y + 0.5, f'{data[y, x]:.{params["precis"]}f}',
                    horizontalalignment='center',
                    verticalalignment='center',
                     fontsize=12
                    )
    #plt.colorbar(heatmap, format=f'%.{precis}f')
    plt.colorbar(heatmap)
    ax.invert_yaxis()    
    plt.xticks( xy_position +0.5, xy_labels)
    plt.yticks( xy_position +0.5, xy_labels)
    plt.tick_params(axis='both', which='major', labelsize=STICK_SIZE, labelbottom = False, bottom=False, top = True, labeltop=True)
    plt.title(f'{params["titleName"]}')

    if params["save"] == True:
        fig.savefig(f'{params["pathOut"]}/{params["fileName"]}.pdf', format='pdf')  

    if params["show"] == False :
        plt.close(fig)        
    elif params["show"] == True:
        #plt.show()
        #plt.gca().set_yscale("log")
        plt.show()          
    del fig  

##===========================================================
#def plotBoxplot(BestTrain, fileName, titleName, unit, dictLay, show=False):
def plotBoxplot(BestTrain, fileName, titleName, **kwargs):
    
    show     = kwargs.get('show')
    save     = kwargs.get('save')
    ext      = kwargs.get('ext')
    unit     = kwargs.get('unit')
    label    = kwargs.get('label')
    dictLay  = kwargs.get('dictLay')
    InputDim = kwargs.get('InputDim')
    
    if label is None: label = 'Layer'

    if show is None: show = False
    if save is None: save = False
    if unit is None: unit = ' '
    if ext  is None: ext  = 'pdf'
        
    nLay  = len(dictLay)                        ## get number of keys to define number of different blocks
    nNets = np.shape(BestTrain)[1]/nLay
    fig   = plt.figure()
    ax    = fig.add_subplot(111)
    n     = [i for i in range(nLay)]
    plt.boxplot(np.sqrt(BestTrain), showfliers=False)
  
    # Add a blue track to separate neural structures
    for i in n[1::2]:
        #plt.axvspan(0.3, nNets+.5, facecolor='b', alpha=0.15)
        plt.axvspan(i*nNets+.5, (1+i)*nNets+.5, facecolor='k', alpha=0.1)

    plt.title(f'{titleName}')
    plt.ylabel(f'rmse {unit}')
    plt.grid(ls='--', lw=0.7)
    #plt.yscale('log')

    # Adjust position and xticks labels
    xticksPosi  = np.linspace(nNets/2, np.shape(BestTrain)[1]-nNets/2, nLay, dtype=int)
    if label.lower() == 'layer':        
        xtickLabels = [f'{n+1}Layer' if n==0 else f'{n+1}Layers' for n in range(nLay)]
    else :        
        xtickLabels = [f'{l}' for l in label]

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.xticks(xticksPosi, xtickLabels)
    #plt.xticks([4,12,20,28],['1Layer', '2Layers', '3Layers', '4Layers']) ## previous method
    if save is True:
        fig.savefig(f'{fileName}.{ext}', format='pdf', dpi=500)  # <<<<<<<<<<<<<<<<<<<<<<<
    if show is True:
        plt.show()
    plt.close(fig)   
    del fig      
        
##===========================================================
def plot_rmse(dictLay, bestFolds, fileName, **kwargs):
              
    show = kwargs.get('show')
    log  = kwargs.get('log')
    ext  = kwargs.get('ext')
    unit = kwargs.get('unit')
              
    if show is None: show = False
    if log  is None: log  = False
    if ext  is None: ext  = 'pdf'
    if unit is None: unit = ' '        

    rmseList = list(np.sqrt(np.mean(bestFolds, axis=1)))
    idxMin   = rmseList.index(np.min(rmseList))
    a = idxMin//dictLay['1']*dictLay['1']
    b = (idxMin//dictLay['1']+1)*dictLay['1'] 
    nLay    = len(dictLay)
    nNets   = int(np.shape(bestFolds)[0]/nLay)
            
    Ndpi = 500
    fig    = plt.figure()
    ax     = fig.add_subplot(111)
    for i in range(len(dictLay)):        
        ax.plot( rmseList[0+i*dictLay['1']: (1+i)*dictLay['1']],'-*',label=f'{i+1}HLayer' if i==0 else f'{i+1}HLayers' )
    
    if log == True:
        ax.set_yscale('log')        
    ax.set_title(f'summary for trainings')    
    xticksPosi  = np.linspace(0, nNets-1, nNets, dtype=int)
    xtickLabels = [f'MLP$_{{{n+1}}}$' for n in range(nNets)]      
    
    ax.grid(ls='--', lw=0.7)
    ax.legend(frameon=False)
    ax.set_ylabel(f'rmse {unit}')
    ax.set_xticks(xticksPosi)
    ax.set_xticklabels(xtickLabels)
    #ax.xaxis.set_major_formatter(plt.NullFormatter())

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    if show == True:
        plt.show()
    fig.savefig(f'{fileName}.{ext}', format=ext, dpi=Ndpi)  # <<<<<<<<<<<<<<<<<<<<<<<
    plt.close(fig)
    del fig

##===========================================================
def plotLoss(history, signal, neurons, path, fileName, log=False, ext='pdf'):

    fig = plt.figure()
    try :
        plt.plot(history.history['loss'], label='train.')
        plt.plot(history.history['val_loss'], label='valid.')
    except :
        plt.plot(history['loss'], label='train.')
        plt.plot(history['val_loss'], label='valid.')

    plt.title(f'Train error {signal} - nNeurons {neurons} ')
    plt.xlabel('epochs')
    plt.ylabel('MSE')
    if log is True:
        plt.yscale('log')
    plt.grid(linestyle='--',linewidth=.7)
    plt.legend(frameon=False)
    fig.savefig(f'{path}{fileName}', format=ext, dpi=700)
    plt.close(fig)    
    
##===========================================================
def plotSS(ssDict, model_type, version, pathSS, **kwargs):
    show = kwargs.get('show')
    save = kwargs.get('save')
                          
    if show is None: show = False
    if save is None: save = False
        
    plt.rcParams["figure.figsize"] = (10,6)
    legend = ['MLP', 'OF', 'REF']
    #save=True
    #show=False

    if model_type.lower() == 'filter':
        for idx, energy in enumerate(eBins):
            print(f'\r creating plots for {energy}...',end='', flush=True)     
            ee = energy.replace('e','').replace('_GV', 'GeV')
            
            ref  = ssDict[energy]['rEta']['E_ref']
            of   = ssDict[energy]['rEta']['E_of']
            pred = ssDict[energy]['rEta']['E_pred']
            RetaRange = ssDict[energy]['rEta']['Reta_xRange']

            n    = int(len(of)/1e3)
            fileName  = f'ss_Reta_{model_type}_{energy}_{n}k_{version}'

            if idx == 0: RetaRange = [0.8, 1.11]
            elif 0 < idx < 4: RetaRange = [0.89, 1.0]
            else: RetaRange = [0.925, 0.98]

            plotHisto(pred, y2=of, y3=ref, pathOut=pathSS, label=f'R$_\eta$', legend=legend, text=ee, fileName=fileName, xRange=RetaRange, save=save, show=show)

            fileName  = f'ss_Rphi_{model_type}_{energy}_{n}k_{version}'
            ref  = ssDict[energy]['rPhi']['E_ref']
            of   = ssDict[energy]['rPhi']['E_of']
            pred = ssDict[energy]['rPhi']['E_pred']    
            RphiRange = ssDict[energy]['rPhi']['Rphi_xRange']

            if idx == 0: RphiRange = [0.68, 1.05]
            elif 0 < idx < 3 : RphiRange = [0.82, 0.985]
            else: RphiRange = [0.9, 0.990]

            plotHisto(pred, y2=of, y3=ref, pathOut=pathSS, label=f'R$_\phi$', legend=legend, text=ee, fileName=fileName, xRange=RphiRange, save=save, show=show)

            fileName  = f'ss_Weta2_{model_type}_{energy}_{n}k_{version}'
            ref  = ssDict[energy]['Weta2']['E_ref']
            of   = ssDict[energy]['Weta2']['E_of']
            pred = ssDict[energy]['Weta2']['E_pred']
            Weta2Range = ssDict[energy]['Weta2']['Weta_xRange']

            if idx == 0: Weta2Range = [1/10000, 2.8/10000]
            elif 0 < idx < 5 : Weta2Range = [1.3/10000, 2.6/10000]
            else: Weta2Range = [1.6/10000, 2.3/10000]

            plotHisto(pred, y2=of, y3=ref, pathOut=pathSS, label=f'W$_{{\eta2}}$', legend=legend, text=ee, fileName=fileName, xRange=Weta2Range, save=save, show=show)
    else:
        for idx, energy in enumerate(eBins):
            print(f'\r creating plots for {energy}...',end='', flush=True)     
            ee = energy.replace('e','').replace('_GV', 'GeV')
            show=False
            save=True

            ref  = ssDict[energy]['rEta']['E_ref']
            of   = ssDict[energy]['rEta']['E_of']
            pred = ssDict[energy]['rEta']['E_pred']
            RetaRange = ssDict[energy]['rEta']['Reta_xRange']

            if idx == 0: RetaRange = [0.758, 1.15]
            elif 0 < idx < 2: RetaRange = [0.875, 1.025]
            else: RetaRange = [0.91, 1.01]

            n    = int(len(of)/1e3)
            fileName  = f'ss_Reta_{model_type}_{energy}_{n}k_{version}'

            plotHisto(pred, y2=of, y3=ref, pathOut=pathSS, label=f'R$_\eta$', legend=legend, text=ee, fileName=fileName, save=save, xRange=RetaRange, show=show, detail=False)

            fileName  = f'ss_Rphi_{model_type}_{energy}_{n}k_{version}'
            ref  = ssDict[energy]['rPhi']['E_ref']
            of   = ssDict[energy]['rPhi']['E_of']
            pred = ssDict[energy]['rPhi']['E_pred'] 
            RphiRange = ssDict[energy]['rPhi']['Rphi_xRange']

            if idx == 0: RphiRange = [0.68, 1.05]
            elif 0 < idx < 3 : RphiRange = [0.82, 0.985]
            else: RphiRange = [0.9, 0.990]

            plotHisto(pred, y2=of, y3=ref, pathOut=pathSS, label=f'R$_\phi$', legend=legend, text=ee, fileName=fileName, xRange=RphiRange, save=save, show=show, detail=False)

            fileName  = f'ss_Weta2_{model_type}_{energy}_{n}k_{version}'
            ref  = ssDict[energy]['Weta2']['E_ref']
            of   = ssDict[energy]['Weta2']['E_of']
            pred = ssDict[energy]['Weta2']['E_pred'] 
            Weta2Range = ssDict[energy]['Weta2']['Weta_xRange']

            if idx == 0: Weta2Range = [1/10000, 2.8/10000]
            elif 0 < idx < 5 : Weta2Range = [1.3/10000, 2.6/10000]
            else: Weta2Range = [1.6/10000, 2.3/10000]

            plotHisto(pred, y2=of, y3=ref, pathOut=pathSS, label=f'W$_{{\eta2}}$', legend=legend, text=ee, fileName=fileName, save=save, xRange=Weta2Range, show=show, detail=False)


    print('\nend ploting!')  
##===========================================================
def r2score(y_true, y_pred, multioutput=False):
    if multioutput == False:
        SS_res =  K.sum(K.square( y_true-y_pred )) 
        SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) ) 
        return ( 1 - SS_res/(SS_tot + K.epsilon()) )
    else :
        #SS_res = np.sum(np.square( y_true-y_pred )) 
        #SS_tot = np.sum(np.square( y_true - np.mean(y_true) ) ) 
        SS_res = np.sum( (y_true - y_pred)**2,axis=0)
        SS_tot = np.sum((y_true - np.mean(y_true,axis=0))**2,axis=0)
        return ( 1 - SS_res/(SS_tot + 1e-8) )

##===========================================================
class ShowerShapes:
    
    def __init__(self, E_ref, E_of, E_pred=None, eta=None ):
        
        self.Reta  = self.Reta()
        self.Rphi  = self.Rphi()
        self.Weta2 = self.Weta2()
        self.E_ref = E_ref
        self.E_of  = E_of
        self.eta   = eta
        self.idx_7x7 = array(range(49))

        if E_pred is None: self.E_pred = np.zeros(E_ref.shape)
        else: self.E_pred = E_pred
    
        self.rEta()
        self.rPhi()
        self.wEta2()
        
    def rEta(self):        

            ## Reta is a paramater that evaluate the ratio: E(3x7).sum()/E(7x7).sum()
            ## First set the most restrict number of clusters (7x7)
            ## Set the clusters size to:size 7x7
            ## This variable is to keep the same set of cluters 7x7 to extract clusters 3x7
        idx3x7 = getIdxClus_mxn( self.idx_7x7.reshape(7,7), 3, 7)        

        self.Reta.E_ref = self.E_ref[:, idx3x7].sum(axis=1)/self.E_ref[:, self.idx_7x7].sum(axis=1)
        self.Reta.E_of  = self.E_of[:, idx3x7].sum(axis=1)/self.E_of[:, self.idx_7x7].sum(axis=1)
        
        #plt.hist(self.E_ref[:, self.idx3x7].sum(axis=1)/self.E_ref[:, self.idx_7x7].sum(axis=1), bins=75)
        #plt.grid(ls='--', lw=0.7)
        #plt.show()
        
        if self.E_pred.sum() == 0: self.Reta_pred = 0
        else: self.Reta.E_pred = self.E_pred[:, idx3x7].sum(axis=1)/self.E_pred[:, self.idx_7x7].sum(axis=1)

        if self.E_of.sum() == 0: self.Reta_of = 0
        else: self.Reta.E_of = self.E_of[:, idx3x7].sum(axis=1)/self.E_of[:, self.idx_7x7].sum(axis=1)

        ## ----------------------
        ## ----------------------
        
    def rPhi(self):        
            ## Rphi is a paramater that evaluate the ratio: E(3x3).sum()/E(3x7).sum()
            ## First set the most restrict number of clusters (3x7)
        idx_3x3 = getIdxClus_mxn( self.idx_7x7.reshape(7,7), 3, 3)        
        idx3x7 = getIdxClus_mxn( self.idx_7x7.reshape(7,7), 3, 7)                

        self.Rphi.E_ref = self.E_ref[:, idx_3x3].sum(axis=1)/self.E_ref[:, idx3x7].sum(axis=1)
        self.Rphi.E_of  = self.E_of[:, idx_3x3].sum(axis=1)/self.E_of[:, idx3x7].sum(axis=1)

        if self.E_pred.sum() == 0: self.Rphi_pred = 0
        else: self.Rphi.E_pred = self.E_pred[:, idx_3x3].sum(axis=1)/self.E_pred[:, idx3x7].sum(axis=1)

        if self.E_of.sum() == 0: self.Rphi_of = 0
        else: self.Rphi.E_of = self.E_of[:, idx_3x3].sum(axis=1)/self.E_of[:, idx3x7].sum(axis=1)

        ## ----------------------
        ## ----------------------
        
    def wEta2(self):
            ## Weta2 is a paramater that evaluate the RMS width for a 3x5 (eta x phi) window around hottest cell        
        idx3x5 = getIdxClus_mxn( self.idx_7x7.reshape(7,7), 3, 5)
        
        Eref_3x5  = self.E_ref[:, idx3x5]
        Eof_3x5   = self.E_of[:, idx3x5]
        Epred_3x5 = self.E_pred[:, idx3x5]
        eta_3x5   = self.eta[:, idx3x5]
        
        self.Weta2.E_ref  = (Eref_3x5*(eta_3x5**2)).sum(axis=1)/Eref_3x5.sum(axis=1) - ((Eref_3x5*eta_3x5).sum(axis=1)/Eref_3x5.sum(axis=1))**2
        self.Weta2.E_of   = (Eof_3x5*(eta_3x5**2)).sum(axis=1)/Eof_3x5.sum(axis=1) - ((Eof_3x5*eta_3x5).sum(axis=1)/Eof_3x5.sum(axis=1))**2
        self.Weta2.E_pred = (Epred_3x5*(eta_3x5**2)).sum(axis=1)/Epred_3x5.sum(axis=1) - ((Epred_3x5*eta_3x5).sum(axis=1)/Epred_3x5.sum(axis=1))**2
        
        ##################################          
    class Reta:
        def __init__(self):
            self.E_ref   = []
            self.E_of    = []
            self.E_of    = []
            self.E_pred  = []
    class Rphi:
        def __init__(self):
            self.E_ref   = []
            self.E_of    = []
            self.E_of    = []
            self.E_pred  = []
    class Weta2:
        def __init__(self):
            self.E_ref   = []
            self.E_of    = []
            self.E_of    = []
            self.E_pred  = []

##===========================================================
def sigmaTauLoss(y_true, y_pred, multioutput=False):
    if multioutput == False:
        SS_res =  K.sum(K.square( y_true-y_pred )) 
        SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) ) 
        return ( 1 - SS_res/(SS_tot + K.epsilon()) )
    else :
        #SS_res = np.sum(np.square( y_true-y_pred )) 
        #SS_tot = np.sum(np.square( y_true - np.mean(y_true) ) ) 
        SS_res = np.sum( (y_true - y_pred)**2,axis=0)
        SS_tot = np.sum((y_true - np.mean(y_true,axis=0))**2,axis=0)
        return ( 1 - SS_res/(SS_tot + 1e-8) )

##===========================================================
def readDirectCells(tree, particle, layer=None, eventMax=None):    
    from math import pi as pi    
    vars = ['e','et','eta','phi']#,'deta','dphi']
    layer_dict = { 
                    0 : 'PSB',
                    1 : 'PSE',
                    2 : 'EMB1',
                    3 : 'EMB2',
                    4 : 'EMB3',
                    5 : 'TileCal1',
                    6 : 'TileCal2',
                    7 : 'TileCal3',
                    8 : 'TileExt1',
                    9 : 'TileExt2',
                    10: 'TileExt3',
                    11: 'EMEC1',
                    12: 'EMEC2',
                    13: 'EMEC3',
                    14: 'HEC1',
                    15: 'HEC2',
                    16: 'HEC3',
    }
    d = { key:[] for key in vars }
    #d['sampling']=[]
    d['hash']    =[]
    d['detector']=[]
    d['layer']       = []
    #d['roi']         = []
    d['eventNumber'] = []
    d['clus_eta']    = []
    d['clus_et']     = []
    d['clus_phi']    = []
    d['clus_eTot']   = []
    d['particle']    = []
    for eventNumber,event in enumerate(tree):
        det_container  = event.CaloDetDescriptorContainer_Cells
        cell_container = event.CaloCellContainer_Cells
        cluster_container = event.CaloClusterContainer_Clusters
        for roi, clus in enumerate(cluster_container):
            for link in clus.cell_links:                
                cell = cell_container[link]
                if layer is None:
                    d['e'].append(cell.e)
                    d['et'].append(cell.et)
                    d['eta'].append(cell.eta)
                    d['phi'].append(cell.phi)
                    d['deta'].append(cell.deta)
                    d['dphi'].append(cell.dphi)
                    d['sampling'].append(det_container[cell.descriptor_link].sampling)
                    d['hash'].append(det_container[cell.descriptor_link].hash)
                    d['detector'].append(det_container[cell.descriptor_link].detector)
                    d['layer'].append(layer_dict[det_container[cell.descriptor_link].sampling])                    
                    d['roi'].append(roi)
                    d['eventNumber'].append(eventNumber)
                    d['clus_et'].append(clus.et)
                    d['clus_eta'].append(clus.eta)
                    d['clus_phi'].append(clus.phi)
                    d['clus_eTot'].append(clus.etot)
                    d['particle'].append(particle)                    
                    
                elif layer_dict[det_container[cell.descriptor_link].sampling] in layer:
                    d['e'].append(cell.e)
                    d['et'].append(cell.et)
                    d['eta'].append(cell.eta)
                    d['phi'].append(cell.phi)
                    #d['deta'].append(cell.deta)
                    #d['dphi'].append(cell.dphi)
                    #d['sampling'].append(det_container[cell.descriptor_link].sampling)
                    d['hash'].append(det_container[cell.descriptor_link].hash)
                    d['detector'].append(det_container[cell.descriptor_link].detector)
                    d['layer'].append(layer_dict[det_container[cell.descriptor_link].sampling])                    
                    #d['roi'].append(roi)
                    d['eventNumber'].append(eventNumber)
                    d['clus_et'].append(clus.et)
                    d['clus_eta'].append(clus.eta)
                    d['clus_phi'].append(clus.phi)
                    d['clus_eTot'].append(clus.etot)
                    d['particle'].append(particle)
        
        if eventMax == None: pass
        elif eventNumber >= eventMax: break
            
    return DataFrame(d)
    
##===========================================================
def regModel(InputDim, OutputDim, nLayers, nNeuron, summary=True):
    model = Sequential()
    model.add(Dense(InputDim, input_dim=InputDim, kernel_initializer='normal', activation='sigmoid'))
    HLneurons = np.linspace(nNeuron, 5, nLayers, dtype='int')

    x = str(np.linspace(InputDim,OutputDim,nLayers+2,dtype=int))
    p=x.replace('[','').replace(']','').replace('  ','_').replace(' ','_')
    if (nLayers > 1):
        for n in range(1,nLayers):
            #model.add(Dense(nNeuron[n], kernel_initializer='normal', activation='relu'))
            model.add(Dense(InputDim, input_dim=InputDim, kernel_initializer='normal', activation=layers.LeakyReLU(alpha=0.3)))
    model.add(Dense(OutputDim, activation='linear'))
    
    desc = f'In{InputDim}_{nLayers}Lay_{nNeuron}Neu_Out{OutputDim}'

    if summary:
        model.summary()
    return model, desc

##########################################
def sizeFile(filename, suffix="B", noName=None):
    num = (os.stat(filename)).st_size
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
        if abs(num) < 1024.0:
            if noName is None:
                return f"{num:.2f} {unit}{suffix}"
            else :
                return f"{filename}: {num:.2f} {unit}{suffix}"
        num /= 1024.0
    if noName is None:
        return f'{num:.2f}Yi{suffix}'
    else :
        return f'{filename}:{num:.2f}Yi{suffix}'

##===========================================================
def plotSigmClus(**kwargs):
    from matplotlib.gridspec import GridSpec
    from matplotlib.ticker import FuncFormatter
    import matplotlib.pyplot as plt
    import numpy as np
    from numpy import array

    # Default parameters
    dictParams = {
        "data": "lzt",
        "day": "",
        "labels": [r'$\sigma_{ref}$', r'$\sigma_{OF}$', r'$\sigma_{NN}$'],
        "log": False,
        "n": "",
        "pathOut": "",
        "sigLastList": None,
        "sigTarg": False,
        "sigPreg": False,
        "sigOF": False,
        "signal": "",
        "save": False,
        "show": True,
        "bar": False,
    }

    # Merge user-provided kwargs with default parameters
    params = {**dictParams, **kwargs}

    # Handle optional parameters
    n = f'_{params["n"]}k' if params["n"] else ""
    day = params["day"]

    # Convert signals to arrays
    sigTarg = array(params["sigTarg"])
    sigPred = array(params["sigPred"])
    sigOF = array(params["sigOF"])
    sigLastList = array(params["sigLastList"]) if params["sigLastList"] is not None else None

    # Initialize figure and axes
    fig = plt.figure(figsize=(10, 6))
    gs = GridSpec(2, 1, figure=fig, height_ratios=[2, 1], hspace=0.05)
    ax1 = fig.add_subplot(gs[0, 0])  # Top plot
    ax2 = fig.add_subplot(gs[1, 0])  # Bottom plot

    # Top subplot: plot signals
    ax1.plot(sigTarg, color="grey", lw=8, alpha=0.4, label=params["labels"][0])
    ax1.plot(sigOF, "^-r", alpha=0.5, label=params["labels"][1])
    ax1.plot(sigPred, "o-", color="blue", alpha=0.5, label=params["labels"][2])

    if sigLastList is not None:
        ax1.plot(sigLastList, "x-", lw=2, color="purple", alpha=0.4, label=params["labels"][3])
        ax2.plot(sigLastList - sigTarg, "x-", color="purple", lw=2, alpha=0.4)

    # Legend and grid for the top subplot
    ax1.legend(frameon=False, ncols=len(params["labels"]))
    ax1.grid(ls="--", lw=0.7)

    # Adjust y-scale and limits
    _, _, _, ymax = ax1.axis()
    if params["log"]:
        ax1.set_yscale("log")
        ax1.set_ylim([0.01, 20 * abs(ymax)])
        if params["signal"].lower() == "energy":
            ax2.set_yscale("log")
        day = f"{day}_log"
    elif len(sigTarg) > 5000:
        ax1.set_ylim([None, 1.4 * ymax])

    # X-axis ticks
    ax1.set_xticks(range(20))
    ax1.set_xticklabels(10 * np.arange(1, 21), rotation=30)

    # Axis labels
    if params["signal"].lower() == "energy":
        #ax1.set_xlabel("E [GeV]")
        ax1.set_ylabel(r"$\frac{\sigma_\hat{E}}{E}$")
    else:
        #ax1.set_xlabel(r"$\hat{\tau}$  [ns]")
        ax1.set_ylabel(r"$\sigma_\hat{\tau}$ [ns]")

    # Bottom subplot: bar or difference plots
    if params["bar"]:
        if params["signal"].lower() == "energy":
            diff = 100 * (sigOF - sigPred) / sigTarg
            ax2.bar(range(20), diff, color="b", lw=2, alpha=0.5)
            ax2.set_ylabel(r"$\Delta\frac{\sigma_\hat{E}}{E}\%$")
            y_min, y_max = np.min(diff), max(diff)
            yticks = np.linspace(y_min, y_max, 4)
            ax2.set_yticks(yticks)
            
            ax2.yaxis.set_major_formatter(FuncFormatter(lambda val, pos: f"{int(val)} \%"))
        else:
            diff = 10 * np.log(sigOF / sigPred)
            ax2.bar(range(20), diff, color="b", lw=2, alpha=0.5)
            #ax2.set_ylabel(r"$\Delta {\hat{\tau}} $")
            ax2.set_ylabel(r"$\sigma_{OF}/\sigma_{NN} $")
        
            # Format y-axis ticks as integers
            y_min, y_max = np.min(diff), max(diff)
            yticks = np.linspace(y_min, y_max, 4)
            ax2.set_yticks(yticks)
            
            ax2.yaxis.set_major_formatter(FuncFormatter(lambda val, pos: f"{int(val)} dB"))
        ax2.grid(ls="--", lw=0.7)
    else:
        ax2.plot(sigOF - sigTarg, "^-", color="r", lw=2, alpha=0.5)
        ax2.plot(sigPred - sigTarg, "o-", color="blue", lw=2, alpha=0.4)
        ax2.grid(ls="--", lw=0.7)

        if params["signal"].lower() == "energy":
            ax2.set_ylabel(r"$\Delta\frac{\sigma_\hat{E}}{E}$")
        else:
            ax2.set_xlabel(r"$\hat{\tau}$  [ns]")
            ax2.set_ylabel(r"$\Delta\sigma_\tau$")
        
    ax1.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    ax2.set_xticks(range(20))
    ax2.set_xticklabels(10 * np.arange(1, 21), rotation=30)
    ax2.set_xlabel("E [GeV]")

    # Show or save the plot
    if params["show"]:
        plt.show()
    if params["save"]:
        import os
        filename = f'sigma_{params["signal"]}_clus_of_{params["data"]}_{day}.pdf'
        filepath = os.path.join(params["pathOut"], filename)
        fig.savefig(filepath, format="pdf")


## ============================================================
def plotSigmCells( **kwargs ):
        
    sigTarg     = kwargs.get('sigTarg')
    sigPred     = kwargs.get('sigPred')
    pathOut     = kwargs.get('pathOut')
    sigOF  = kwargs.get('sigOF')
    signal = kwargs.get('signal')
    day    = kwargs.get('day')
    show   = kwargs.get('show')
    log    = kwargs.get('log')
    save   = kwargs.get('save')    
    n      = kwargs.get('n')    
        
    if show is None: show=True
    if signal is None: signal=True
    if save is None: save=False
    if log  is None: log=False
        
    ij_cell = ['-3,3' , '-2,3' , '-1,3' , '0,3' , '1,3' , '2,3' , '3,3' , 
           '-3,2' , '-2,2' , '-1,2' , '0,2' , '1,2' , '2,2' , '3,2' , 
           '-3,1' , '-2,1' , '-1,1' , '0,1' , '1,1' , '2,1' , '3,1' , 
           '-3,0' , '-2,0' , '-1,0' , '0,0' , '1,0' , '2,0' , '3,0' , 
           '-3,-1', '-2,-1', '-1,-1', '0,-1', '1,-1', '2,-1', '3,-1', 
           '-3,-2', '-2,-2', '-1,-2', '0,-2', '1,-2', '2,-2', '3,-2', 
           '-3,-3', '-2,-3', '-1,-3', '0,-3', '1,-3', '2,-3', '3,-3' ]
    idx_7x7 = array(range(49))
    idx_5x5 = getIdxClus_mxn(idx_7x7.reshape(7,7), 5, 5)

    sigTarg  = array(sigTarg)
    sigPreds = array(sigPred)
    #sigRaw  = array(sigRawCells)
    sigOF    = array(sigOF)
    if signal.lower() == 'energy': label_ET = 'E'
    else: label_ET = 'T'
    index = array(range(sigOF.shape[0] ))
    
    for i in range(25):
        fig = plt.figure(figsize=(10,6)) 
        gs = GridSpec(2, 1, figure=fig, height_ratios=[2, 1], hspace=0)  # Set the height ratios and eliminate vertical space

        ax1 = fig.add_subplot(gs[0, 0])
        #ax1.set_title('Top Subplot')
        # Create the bottom subplot
        ax2 = fig.add_subplot(gs[1, 0])

        ax1.plot(sigTarg[:,i], color='grey', lw=8, alpha=0.4, label=r'$E_{ref}$')
        #if signal.lower() == 'energy':
            #ax1.plot(sigRawCells[:,i], 'x-', lw=2, color='k', alpha=0.4, label=r'$E_{ref}+XT+N_{oise}$')

        ax1.plot(sigOF[:,i], '^-r', alpha=0.5, label=r'OF')
        ax1.plot(sigPred[:,i], 'o-', color='blue', alpha=0.5, label=r'$E_{mlp}$')
        ax1.legend(frameon=False)
        ax1.grid( ls='--',lw=.7)
        
        xmin, xmax, ymin, ymax = ax1.axis()
        if log == True:
            ax1.set_yscale('log')
            ax1.set_ylim([0.01, 20*abs(ymax)])
        elif len(sigTarg) > 5e3:        
            ax1.set_ylim([None, 1.4*ymax])        

        ax1.set_xticks( index, 10*array(range(1,21)), rotation=30 )
        #ax1.set_xlabel('E  [GeV]')

        ax2.plot( sigOF[:,i]- sigTarg[:,i], '^-', color='r', lw=2, alpha=0.5)
        ax2.plot( sigPred[:,i]- sigTarg[:,i], 'o-', color='blue', lw=2, alpha=0.4)    
        ax2.grid( ls='--',lw=.7)
        ax2.set_xticks( index, 10*array(range(1,21)), rotation=30 )
        
        #if log is True:
            #ax2.set_yscale('log')
        if signal.lower() == 'energy':
            #ax2.set_xlabel(f'E [GeV] - Cell {ij_cell[idx_5x5[i]]}')
            ax1.set_ylabel(r'$\frac{\sigma_E}{E}$')
            ax2.set_ylabel(r'$\Delta\frac{\sigma_E}{E}$ [GeV]')
        else:            
            #ax2.set_xlabel(r'$\tau$  [ns]'+f' - Cell {ij_cell[idx_5x5[i]]}')
            ax1.set_ylabel(r'$\sigma_\tau$')
            ax2.set_ylabel(r'$\Delta\sigma_\tau$ [ns]')        
        ax2.set_xlabel(f'E [GeV] - Cell {ij_cell[idx_5x5[i]]}')
        if show is True:
            plt.show()
        #fig.savefig(f'{pathPKL}/sigmas/sigma_E_cluster_of_cell_{ij_cell[idx_5x5[i]]}.pdf', format='pdf')
        if save is True:
            fig.savefig(f'{pathOut}/sigma_{label_ET}_of_cell_{ij_cell[idx_5x5[i]]}_{n}k_{day}.pdf', format='pdf')
            #fig.savefig(f'{pathSigma}/sigma_E_cluster_of_cell_{ij_cell[idx_5x5[i]]}.pdf', format='pdf')

##===========================================================
## Plot signal 
def plotSignal(pathOut, label, legend, fileName, titleName, x1, x2=None, **kwargs): 
    
    ext  = kwargs.get('ext')
    show = kwargs.get('show')
    log  = kwargs.get('log')
    save = kwargs.get('save')
                      
    if ext  is None: ext  = 'pdf'
    if show is None: show = False
    if log  is None: log  = False
    if save is None: save = False

    
    Ndpi = 700
    fig  = plt.figure()
    ax   = fig.add_subplot(111)
    yLabel = 'Amp Norm'
    
    if x2 is not None:        
        if not legend :
            plt.plot(x1, color='k')
            plt.plot(x2, color='k')
        else :
            plt.plot(x1, color='k', label=f'{legend[0]}')
            plt.plot(x2, color='k', label=f'{legend[1]}')
            plt.legend(frameon=False)
    else :
        if not legend :
            plt.plot(x1, color='k')
        else :
            plt.plot(x1,  color='k', label=f'{legend}')            
            plt.legend(frameon=False)

    ax.set_title(titleName)
    ax.set_xlabel(f'{label}')
    ax.set_ylabel(f'{yLabel}')
    ax.grid(linestyle='--',linewidth=.7)

    if save == True:
        fig.savefig(f'{pathOut}/{fileName}', format=ext, dpi=Ndpi)
    #print(f'Show plot: {show}')

    if log == True:
        plt.gca().set_yscale("log")

    if (show == False ):
        plt.close(fig) 
    elif (show == True):        
        #plt.show()
        #plt.gca().set_yscale("log")
        plt.show()

## ==========================================================
def plotHisto(y1, **kwargs):
    """
    pathOut   - where you want to store the plot
    label     - is a list with 4 informations: 1st, 2nd, 3rd is the label for the signal, and the 4th is the legend position
    fileName  - File name to store
    titleName - Plot title
    show      - True sows the plot, False none
    log       - True ylog scale, False none
    ext       - File extension
    MeanRms:  
              - 'mean' to calculate mean and std values for each signal => mean +/- std
              - 'rms' to calculate the rms and rmse values for each signal => rms +/- rmse
    save      - True save plot on pathOut, false none
    y1        - first signal  (mlp)
    y2        - second signal (of)
    y3        - third signal (target)
    """    
    # Define default values for parameters
    default_params = {
        "axisFormat": False,        
        "adjustXlim": False,
        "cell_00": False,
        "colLegend": False,
        "color": False,
        "detail": False,
        "ext": "pdf",
        "fileName": "",
        "label": "",
        "legend": None,
        "log": False,
        "MeanRms": "mean",
        "nStd": None,
        "pathOut": "",
        "save": False,        
        "show": True,
        "titleName": "",
        "text": [],
        "unit": "",
        "xRange": None,        
        "y2": None,
        "y3": None,
        "ss": False,
        "ks": None
        }
    
    def custom_formatter(x, pos, scale_x):
        """
        Custom tick formatter for the x-axis.
        """
        if scale_x != 1:
            magnitude = len(str(int(x))) - 1  # Calculate the magnitude of the number
            decimals = max(0, 2 - magnitude)   # Determine the number of decimal places
            return f'{x * scale_x:.2f}'
        else:
            return f'{x:.2f}'
            
        # Update default parameters with provided keyword arguments
    params = {**default_params, **kwargs}

    if params["color"]:
        listColors = ['royalblue', 'sandybrown', 'gainsboro']
        listEdges  = ['navy', 'orangered', 'black']
        #listColors = ['khaki', 'purple', 'gainsboro']
        #listEdges  = ['darkorange', 'indigo', 'black']
    else:
        listColors = ['royalblue', 'indianred', 'gainsboro']
        listEdges  = ['navy', 'red', 'black']
        
    plt.rcParams["figure.figsize"] = (10,6)
    scale_x = 1    
    xscale  = 1        
    y2, y3 = params["y2"], params["y3"]

    if y2 is not None and y3 is not None:
        arrays = [y1, y2, y3]    
    elif y2 is not None:                
        arrays = [y1, y2]
    else:            
        arrays = [y1]
        
    if params["label"].lower() == 'energy':
        if rms( array( arrays ) ) < 0.01:
            arrays = list(10000*array(arrays))
            xLabel = r'Energy [$10^{-3}$'+f' {params["unit"]}]'
        else :
            xLabel = f'Energy [{params["unit"]}]'
    elif params["label"] == 'time':        
        xLabel = 'Time [ns]'
        unit   = 'ns'        
    elif params["label"] == 'ss':
            xscale = 1
            xLabel = params["label"]#label
            unit   = ''        
    else :
        if rms( array(arrays)) < 0.01:
            xscale  = 10000
            xLabel = f'{params["label"]} '+r'[$10^{-4}$]'            
        else: 
            xscale = 1
            xLabel = params["label"]#label
            unit   = ''                
    if params["ss"]:
        xscale = 100
        xLabel = f'{params["label"]} '+r'[$10^{-2}$]'   
        
    Nbins, Ndpi  = 100, 700
    if params["xRange"]:        
        minBin, maxBin = min(params["xRange"]), max(params["xRange"])
    else :
        minBin, maxBin = min_max( arrays )

    if params["adjustXlim"]:
        if params["cell_00"]: pass
        else:
            if params["nStd"] is None:
                nStd = 3
            else:
                nStd = params["nStd"]

            if np.min(arrays)>0 and not params["cell_00"]:
                nn = 3
            else: nn = 1
            
            minBin, maxBin = np.mean(arrays) - nStd/nn*np.std(arrays), np.mean(arrays) + nStd*np.std(arrays)
        #nBins = np.linspace(xmin, xmax, 100)
        
    ## To adjust time displayment to better visualization
    ## once the time expected is a zero mean distribution
    valueBins = np.linspace(minBin, maxBin, Nbins)
    
    ## ===========================
    ## ---------- Details --------    
    unit    = params["unit"]
    if params["detail"]:
        #arrays = [y1, y2, y3]       
        k_NN,  p_NN  = ks_2samp(arrays[0], arrays[1])        
        
        fig, ax = plt.subplots(1, 2, gridspec_kw={'width_ratios': [10, 5]})
        plt.subplots_adjust(wspace=0.05) 
        maxStd = maxBin
        meanRmsResults = getMeanRms( arrays , MeanRms=params["MeanRms"])
        minBin, maxBin = min_max( arrays )
        valueBins = np.linspace(minBin, maxBin, Nbins)
        
        #print(meanRmsResults)
        
        ## Left plot: Zoomed in version
        for i, arr in enumerate(arrays):
            n_arr = len(arrays)
            label = params["legend"][i] #if params["legend"] and i < len(params["legend"]) #else f'Data {i+1}'
            color, edgecolor, alpha = ['royalblue', 'indianred', 'gainsboro'][i % n_arr], ['navy', 'red', 'black'][i % n_arr], [0.5, 0.5, 0.4][i % n_arr]
            ax[0].hist(arr, bins=valueBins, ec=edgecolor, alpha=alpha, fc=color, lw=1.2, histtype='stepfilled', label=f'{label}'.ljust(10, ' ') + f': {meanRmsResults[i][0]*xscale:.2f} $\pm$ {meanRmsResults[i][1]*xscale:.2f}'.rjust(22, ' '))
            
        xmin,xmax,ymin,ymax = ax[0].axis()
        ax[0].legend(frameon=False, loc='upper right')

        if params["log"]:
            ax[0].set_yscale('log')
            ax[0].set_ylim([10, 20*abs(ymax)])
        elif len(y1) >= 1e3:
            scale_y = 1e3
            ax[0].set_ylim([None, 1.4*ymax])
            ax[0].set_ylabel(f'Count'+r' [10$^3$]')    
            
            ticks_y = ticker.FuncFormatter(lambda y1, pos: '{0:g}'.format(y1/scale_y))
            ax[0].yaxis.set_major_formatter(ticks_y)
        else: ax[0].set_ylabel('Count')            

        if len(params["text"]) == 2:
            textstr = '\n'.join((
            f'{params["text"][0]}',
            f'{params["text"][1]}'))
            ax[0].text(0.05, 0.65, textstr, transform=ax[0].transAxes, fontsize=16,
            verticalalignment='top')
        else:
            textstr = f'\n{params["text"][0]}'
            ax[0].text(0.05, 0.7, textstr, transform=ax[0].transAxes, fontsize=16,
            verticalalignment='top')   
            
        ax[0].set_xlabel(f'{xLabel}')         
        ax[0].grid(ls='--', lw=0.65)
        ax[0].spines['top'].set_visible(False)
        ax[0].spines['right'].set_visible(False)

        ## ==========================
        ## Right plot: Zoomed in version
        del params["legend"][1]
        del arrays[1]
        minBin, maxBin = min_max( arrays )
        valueBins = np.linspace(minBin, maxBin, Nbins)

        for i, arr in enumerate(arrays):
            label = params["legend"][i] #if params["legend"] and i < len( params["legend"] ) #else f'Data {i+1}'
            color, edgecolor, alpha = ['royalblue', 'gainsboro'][i % 2], ['navy', 'black'][i % 2], [0.6, 0.6][i % 2]
            ax[1].hist(arr, bins=valueBins, ec=edgecolor, alpha=alpha, fc=color, lw=1.3, histtype='stepfilled', label=label, )
                
        # Set zoomed-in x-axis limits if adjustXlim is set
        n, bins, _ = ax[1].hist(arrays[0], bins=valueBins, alpha=0)  # Just to get bin counts without drawing
        
        xmin,xmax,ymin,ymax = ax[1].axis()
       
        if params["adjustXlim"]:
            nn = np.where(n>10)
            if len(nn[0]) < 40:
                ii = list(n).index(np.max(n))
                if ii+5 > 99:
                    ax[1].set_xlim([bins[ii-5], bins[-1]])
                elif ii-5 < 0:
                    ax[1].set_xlim([bins[0], bins[ii+5]])
                else :
                    ax[1].set_xlim([bins[ii-5], bins[ii+5]])
        
        ax[1].legend(frameon=False, loc='upper right')
        #ax[1].yaxis.set_major_formatter(plt.NullFormatter())

        if params["log"]:
            ax[1].set_yscale('log')
            ax[1].set_ylim([10, 30*abs(ymax)])
        elif len(y1) >= 1e3:
            scale_y = 1e3
            ax[1].set_ylim([None, 1.4*ymax])
            ax[1].set_ylabel('')  # No ylabel
            # Add y-axis ticks on the right for ax[1]
            ax[1].yaxis.tick_right()  # Moves the y-axis ticks to the right
            ax[1].tick_params(axis='y', which='both', left=False, right=True, labelleft=False, labelright=True)  # Enable right ticks, disable left ticks
            # Add ylabel on the right for ax[1]
            ax[1].yaxis.set_label_position('right')  # Move the ylabel to the right
            ax[1].set_ylabel(f'Count'+r' [10$^3$]') # Add ylabel text
            
            ticks_y = ticker.FuncFormatter(lambda y1, pos: '{0:g}'.format(y1/scale_y))
            ax[1].yaxis.set_major_formatter(ticks_y)
            
        else:            
            ax[1].set_ylim([None, 1.4*ymax])
            ax[1].set_ylabel('')  # No ylabel
            # Add y-axis ticks on the right for ax[1]
            ax[1].yaxis.tick_right()  # Moves the y-axis ticks to the right
            ax[1].tick_params(axis='y', which='both', left=False, right=True, labelleft=False, labelright=True)  # Enable right ticks, disable left ticks
            # Add ylabel on the right for ax[1]
            ax[1].yaxis.set_label_position('right')  # Move the ylabel to the right
            ax[1].set_ylabel('Count')                        
        ax[1].set_xlim([ np.mean(arrays) - 3*np.std(arrays), np.mean(arrays) + 3*np.std(arrays) ])            
        ax[1].grid(ls='--', lw=0.7)
        ax[1].set_xlabel(f'{xLabel}') 
        ax[1].spines['top'].set_visible(False)
        ax[1].spines['right'].set_visible(False)        

                # Join all items with line breaks
        textstr = f"KS: {params['ks']:.3f}"

        # Add the text to the plot
        ax[1].text(0.05, 0.75, textstr, transform=ax[1].transAxes, fontsize=18,
                verticalalignment='top') 
    ## ===========================
    ## ---- Whithout Details -----
    else :        
        meanRmsResults = getMeanRms(arrays, MeanRms=params["MeanRms"])
        fig    = plt.figure()
        ax     = fig.add_subplot(111)
        MeanRms = params["MeanRms"]
        titleName = params["titleName"]
        
        for i, arr in enumerate(arrays):
            n_arr = len(arrays)
            label = params["legend"][i] #if params["legend"] and i < len(params["legend"]) #else f'Data {i+1}'
            #color, edgecolor, alpha = ['royalblue', 'indianred', 'gainsboro'][i % n_arr], ['navy', 'red', 'black'][i % n_arr], [0.6, 0.5, 0.5][i % n_arr]
            color, edgecolor, alpha = listColors[i % n_arr], listEdges[i % n_arr], [0.6, 0.5, 0.5][i % n_arr]
            ax.hist(arr, bins=valueBins, ec=edgecolor, alpha=alpha, fc=color, lw=1.2, histtype='stepfilled', label=f'{label}'.ljust(10, ' ') + f': {meanRmsResults[i][0]*xscale:.2f} $\pm$ {meanRmsResults[i][1]*xscale:.2f}'.rjust(22, ' '))

        n, bins, _ = ax.hist(arrays[0], bins=valueBins, alpha=0)  # Just to get bin counts without drawing
        
        ax.set_title( params["titleName"])
        ax.set_xlabel(f'{xLabel}')
        ax.grid(linestyle='--',linewidth=.7)
        ax.legend(frameon=False, loc=1)

        xmin,xmax,ymin,ymax = ax.axis()        
        if params["log"]:
            plt.gca().set_yscale("log")
            ymax = max(set(n))
            ax.set_ylim([10, 40*abs(ymax)])
            ax.set_ylabel(f'Count')#+r' [10$^3$]')        
            scale_y = 1e3
            fileName = f'{params["fileName"]}_log'
        elif len(y1) > 5e3:
            scale_y = 1e3
            ax.set_ylim([None, 1.5*ymax])
            ax.set_ylabel(f'Count'+r' [10$^3$]')
        else :
            scale_y = 1
            ax.set_ylim([None, 1.5*ymax])
            ax.set_ylabel(f'Count')
            
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ticks_y = ticker.FuncFormatter(lambda y1, pos: '{0:g}'.format(y1/scale_y))
        ax.yaxis.set_major_formatter(ticks_y)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: custom_formatter(x, pos, xscale)))
    if not params["detail"]:
        
        # Insert an extra space after the first item
        #if len(params["text"]) > 1:
            #params["text"].insert(1, '')  # Adds an extra blank line after the first item
        # Join all items with line breaks
        textstr = '\n'.join(params["text"])
        if params["ks"] is not None:
            textstr += f"\nKS: {params['ks']:.3f}"
        # Add the text to the plot
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=18,
                verticalalignment='top')  

    if params["save"]:
        fig.savefig(f'{params["pathOut"]}/{params["fileName"]}.{params["ext"]}', format='pdf', dpi=Ndpi)    

    if not params["show"]:
        plt.close(fig)
        plt.close()
    elif params["show"]:
        plt.show()
    del fig     

## ==========================================================
def old_plotHisto(y1, y2=None, y3=None, **kwargs):
    """
    pathOut   - where you want to store the plot
    label     - is a list with 4 informations: 1st, 2nd, 3rd is the label for the signal, and the 4th is the legend position
    fileName  - File name to store
    titleName - Plot title
    show      - True sows the plot, False none
    log       - True ylog scale, False none
    ext       - File extension
    MeanRms:  
              - 'mean' to calculate mean and std values for each signal => mean +/- std
              - 'rms' to calculate the rms and rmse values for each signal => rms +/- rmse
    save      - True save plot on pathOut, false none
    y1        - first signal  (mlp)
    y2        - second signal (of)
    y3        - third signal (target)
    
    """    

    # Define default values for parameters
    default_params = {
        "axisFormat": False,        
        "adjustXlim": False,
        "detail": False,
        "ext": "pdf",
        "fileName": "",
        "label": "",
        "legend": ['','',''],
        "log": False,
        "MeanRms": "mean",
        "pathOut": "",
        "save": False,        
        "show": True,
        "titleName": "",
        "unit": "GeV",
        "xRange": None,   
    }
    def custom_formatter(x, pos):
        threshold = 0.001  # Threshold for scaling
        scale_x = 10000
        if x < threshold:
            return '{:.3f}'.format(x * scale_x)  # Format the scaled value to 3 decimal places
        else:
            magnitude = len(str(int(x))) - 1  # Calculate the magnitude of the number
            decimals = max(0, 3 - magnitude)   # Determine the number of decimal places
            return '{:.{dec}f}'.format(x, dec=decimals)  # Format with the determined number of decimal places
        # Update default parameters with provided keyword arguments
    params = {**default_params, **kwargs}

    plt.rcParams["figure.figsize"] = (10,6)
    scale_x = 1    
    
    #if label.lower() == 'energy':
    if params["label"] == 'energy':
        if y2 is not None and y3 is not None: 
            if rms(y1) < 0.01 or rms(y2) < 0.01 or rms(y3) < 0.01:
                y1, y2, y3 = y1*1e3, y2*1e3, y3*1e3
                xLabel = r'Energy [$10^{-3}$'+f' {unit}]'
                #unit   = 'MeV'
            else :
                xLabel = f'Energy [{unit}]'
                #unit   = 'GeV'
        elif y2 is not None:
            if rms(y1) < 0.01 or rms(y2) < 0.01:
                y1, y2 = y1*1e3, y2*1e3                
                xLabel = r'Energy [$10^{-3}$'+f' {unit}]'
                #unit   = 'MeV'
            else :
                xLabel = f'Energy [{unit}]'
                #unit   = 'GeV'
        else:
            if rms(y1) < 0.01:
                y1 = y1*1e3
                xLabel = r'Energy [$10^{-3}$'+f' {unit}]'
                #unit   = 'MeV'
            else :
                xLabel = f'Energy [{unit}]'
                #unit   = 'GeV'
    #elif label.lower() == 'time':
    elif params["label"] == 'time':        
        xLabel = 'Time [ns]'
        unit   = 'ns'
    #elif label.lower() == 'ss':
    elif params["label"] == 'ss':
            xscale = 1
            xLabel = params["label"]#label
            unit   = ''        
    else :
        if rms(y1) < 0.01 or rms(y2) < 0.01 or rms(y3) < 0.01:
            #if xscale  is None: 
            xscale  = 10000
            xLabel = f'{params["label"]} '+r'[$10^{-4}$]'  
        else: 
            xscale = 1
            xLabel = params["label"]#label
            unit   = ''                
        
    Nbins, Ndpi  = 100, 700
    #if xRange is not None:
    if params["xRange"]:        
        minBin, maxBin = min(params["xRange"]), max(params["xRange"])
    else :
        minBin, maxBin = min_max(y1, y2, y3)
    valueBins = np.linspace(minBin, maxBin, Nbins)
    
    ## ===========================
    ## ---------- Details --------
    #if detail is True:
    if params["detail"]:
        fig, ax = plt.subplots(1, 2, gridspec_kw={'width_ratios': [10, 6]})
        maxStd = maxBin
        meanMean = np.mean([y1, y2, y3])
        y1MeanRms, y2MeanRms, y3MeanRms = getMeanRms(y1, x2=y2, x3=y3, MeanRms=MeanRms)
        ## ==========================
        ## ==============
        ## THREE plots
        #n, bins, patches = ax.hist(y1,  bins=valueBins,  ec='navy',       alpha=0.7, fc='royalblue', lw=1.2, histtype='stepfilled', label=f'{legend[0]}'.ljust(14,' ')+f'rms: {rms(y1):.2f} $\pm$ {rmsErr(y1):.2f} {unit}'.ljust(28,' '))
        n, bins, patches = ax[0].hist(y1,  bins=valueBins,  ec='navy', alpha=0.7, fc='royalblue', lw=1.2, histtype='stepfilled', label=f'{legend[0]}'.ljust(14,' ')+ f': {y1MeanRms[0]*xscale:.2f} $\pm$ {y1MeanRms[1]*xscale:.2f} '.ljust(20,' '))
        ax[0].hist(y2,  bins=valueBins,  ec='red',  alpha=0.5, fc='indianred', lw=1.2, histtype='stepfilled', label=f'{legend[1]}'.ljust(14,' ')+ f': {y2MeanRms[0]*xscale:.2f} $\pm$ {y2MeanRms[1]*xscale:.2f} '.ljust(20,' '))
        ax[0].hist(y3,  bins=valueBins,  ec='k',    alpha=0.4, fc='gainsboro', lw=1.8, histtype='stepfilled', label=f'{legend[2]}'.ljust(14,' ')+ f': {y3MeanRms[0]*xscale:.2f} $\pm$ {y3MeanRms[1]*xscale:.2f} '.ljust(20,' '))
    
        xmin,xmax,ymin,ymax = ax[0].axis()
        ax[0].legend(frameon=False)

        if log == True:
            ax[0].set_yscale('log')
            ax[0].set_ylim([10, 20*abs(ymax)])
        elif len(y1) > 5e3:
            scale_y = 1e3
            ax[0].set_ylim([None, 1.4*ymax])
            ax[0].set_ylabel(f'Count'+r' [10$^3$]')    
            
            ticks_y = ticker.FuncFormatter(lambda y1, pos: '{0:g}'.format(y1/scale_y))
            ax[0].yaxis.set_major_formatter(ticks_y)
        else: ax[0].set_ylabel('Count')            
            
        ax[0].set_xlabel(f'{xLabel}')         
        ax[0].grid(ls='--', lw=0.7)
        ax[0].spines['top'].set_visible(False)
        ax[0].spines['right'].set_visible(False)
        
        if adjustXlim is True:
            if len(nn[0]) < 40:
                ii = list(n).index(np.max(n))
                if ii+5 > 99:
                    ax[0].set_xlim([bins[ii-5], bins[-1]])
                elif ii-5 < 0:
                    ax[0].set_xlim([bins[0], bins[ii+5]])
                else :
                    ax[0].set_xlim([bins[ii-5], bins[ii+5]])

        n, bins, patches = ax[1].hist(y1,  bins=valueBins,  ec='navy', alpha=0.7, fc='royalblue', lw=1.2, histtype='stepfilled', label=f'{legend[0]}')
        ax[1].hist(y3,  bins=valueBins,  ec='k',    alpha=0.5, fc='gainsboro', lw=1.6, histtype='stepfilled', label=f'{legend[2]}')
        xmin,xmax,ymin,ymax = ax[1].axis()
        
        nn = np.where(n>10)
        #if adjustXlim is True:
        if params["adjustXlim"]:
            if len(nn[0]) < 40:
                ii = list(n).index(np.max(n))
                if ii+5 > 99:
                    ax[1].set_xlim([bins[ii-5], bins[-1]])
                elif ii-5 < 0:
                    ax[1].set_xlim([bins[0], bins[ii+5]])
                else :
                    ax[1].set_xlim([bins[ii-5], bins[ii+5]])
        
        ax[1].legend(frameon=False)
        ax[1].yaxis.set_major_formatter(plt.NullFormatter())
        #if log == True:
        if params["log"]:            
            ax[1].set_yscale('log')
            ax[1].set_ylim([10, 20*abs(ymax)])
        elif len(y1) > 5e3:
            scale_y = 1e3
            ax[1].set_ylim([None, 1.4*ymax])
            ticks_y = ticker.FuncFormatter(lambda y1, pos: '{0:g}'.format(y1/scale_y))
            ax[1].yaxis.set_major_formatter(ticks_y)
            
        #ax[1].axes.yaxis.set_visible(False)
        ax[1].grid(ls='--', lw=0.7)
        ax[1].set_xlabel(f'{xLabel}') 
        ax[1].spines['top'].set_visible(False)
        ax[1].spines['right'].set_visible(False)        
             
    ## ===========================
    ## ---- Whithout Details -----
    else :        
        fig    = plt.figure()
        ax     = fig.add_subplot(111)
        MeanRms = params["MeanRms"]
        unit    = params["unit"]
        titleName = params["titleName"]
        if (y2 is not None and y3 is not None):
            
            y1MeanRms, y2MeanRms, y3MeanRms = getMeanRms(y1, x2=y2, x3=y3, MeanRms=MeanRms)
            maxStd = max(np.std(y1), np.std(y2), np.std(y3))
            meanMean = np.mean([y1, y2, y3])            
            
            n, bins, patches = ax.hist(y1,  bins=valueBins,  ec='navy',       alpha=0.7, fc='royalblue', lw=1.2, histtype='stepfilled', label=f'{legend[0]}'.ljust(14,' ')+ f'{MeanRms}: {y1MeanRms[0]*xscale:.2f} $\pm$ {y1MeanRms[1]*xscale:.1e} {unit}'.ljust(28,' '))
            #ax.hist(y1,  bins=valueBins,  ec='navy', alpha=0.7, fc='royalblue', lw=1.2, histtype='stepfilled', label=f'{legend[0]}'.ljust(14,' ')+ f'{MeanRms}: {y1MeanRms:.2f} $\pm$ {y1StdRmse:.2f} {unit}'.ljust(28,' '))
            ax.hist(y2,  bins=valueBins,  ec='red',  alpha=0.6, fc='indianred', lw=1.2, histtype='stepfilled', label=f'{legend[1]}'.ljust(14,' ')+ f'{MeanRms}: {y2MeanRms[0]*xscale:.2f} $\pm$ {y2MeanRms[1]*xscale:.1e} {unit}'.ljust(28,' '))            
            ax.hist(y3,  bins=valueBins,  ec='k',    alpha=0.5, fc='gainsboro', lw=1.5, histtype='stepfilled', label=f'{legend[2]}'.ljust(14,' ')+ f'{MeanRms}: {y3MeanRms[0]*xscale:.2f} $\pm$ {y3MeanRms[1]*xscale:.1e} {unit}'.ljust(28,' '))

            nn = np.where(n>10)
                ## --------------- ##
        ## ==========================
        ## ==============
        ## TWO plots
        elif (y2 is not None):
            y1MeanRms, y2MeanRms = getMeanRms(y1, x2=y2, MeanRms=MeanRms)
            maxStd = max(np.std(y1), np.std(y2))
            meanMean = np.mean([y1, y2])        

            n, bins, patches = ax.hist(y1,  bins=valueBins,  ec='navy', alpha=0.7, fc='royalblue', lw=1.2, histtype='stepfilled', label=f'{legend[0]}'.ljust(14,' ')+ f'{MeanRms}: {y1MeanRms[0]:.2f} $\pm$ {y1MeanRms[1]:.2f} {unit}'.ljust(28,' '))            
            ax.hist(y2,  bins=valueBins,  ec='red',  alpha=0.6, fc='indianred', lw=1.2, histtype='stepfilled', label=f'{legend[1]}'.ljust(14,' ')+ f'{MeanRms}: {y2MeanRms[0]:.2f} $\pm$ {y2MeanRms[1]:.2f} {unit}'.ljust(28,' '))            

        ## ==========================
        ## ==============
        ## ONE plot
        else :
            y1MeanRms = getMeanRms(y1, MeanRms=MeanRms)
            maxStd = np.std(y1)
            meanMean = np.mean(y1)
            n, bins, patches = ax.hist(y1,  bins=valueBins,  ec='k',    alpha=0.5, fc='gainsboro', lw=1.7, histtype='stepfilled', label=f'{legend[0]}'.ljust(14,' ')+ f'{MeanRms}: {y1MeanRms[0]:.2f} $\pm$ {y1MeanRms[1]:.2f} {unit}'.ljust(28,' '))
        
        ax.set_title(titleName)
        ax.set_xlabel(f'{xLabel}')
        ax.grid(linestyle='--',linewidth=.7)
        #ax.legend(frameon=False, loc=legend[3])
        ax.legend(frameon=False)

        xmin,xmax,ymin,ymax = ax.axis()
        nn = np.where(n>5)
        #if log == True:
        if params["log"]:
            plt.gca().set_yscale("log")
            #ax.set_ylim([None, 40*abs(ymax)])
            ax.set_ylim([100, 8*abs(ymax)])
            ax.set_ylabel(f'Count'+r' [10$^3$]')        
            scale_y = 1e3
            fileName = f'{fileName}_log'
        elif len(y1) > 5e3:
            scale_y = 1e3
            ax.set_ylim([None, 1.4*ymax])
            ax.set_ylabel(f'Count'+r' [10$^3$]')
        else :
            scale_y = 1
            ax.set_ylim([None, 1.4*ymax])
            ax.set_ylabel(f'Count')
            
        limit = 0.0
        #if adjustXlim is True:
        if params["adjustXlim"]:
            if len(nn[0]) < 40:
                ii = list(n).index(np.max(n))
                if ii+5 > 99:
                    ax.set_xlim([bins[ii-5], bins[-1]])
                elif ii-5 < 0:
                    ax.set_xlim([bins[0], bins[ii+5]])
                else :
                    ax.set_xlim([bins[ii-5], bins[ii+5]])
            
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ticks_y = ticker.FuncFormatter(lambda y1, pos: '{0:g}'.format(y1/scale_y))
        ax.yaxis.set_major_formatter(ticks_y)
        ax.xaxis.set_major_formatter(FuncFormatter(custom_formatter))

    if params["save"]:
        fig.savefig(f'{params["pathOut"]}/{params["fileName"]}.{params["ext"]}', format=ext, dpi=Ndpi)
    #print(f'Show plot: {show}')

    if not params["show"]:
    #if show == False :
        plt.close(fig)
        plt.close()
    #elif show == True:
    elif params["show"]:
        #plt.show()
        #plt.gca().set_yscale("log")
        plt.show()
    del fig  
    
## ==========================================================
#@profile
def __plotHisto(y1, y2=None, y3=None, **kwargs):
    """
    pathOut   - where you want to store the plot
    label     - is a list with 4 informations: 1st, 2nd, 3rd is the label for the signal, and the 4th is the legend position
    fileName  - File name to store
    titleName - Plot title
    show      - True sows the plot, False none
    log       - True ylog scale, False none
    ext       - File extension
    MeanRms:  
              - 'mean' to calculate mean and std values for each signal => mean +/- std
              - 'rms' to calculate the rms and rmse values for each signal => rms +/- rmse
    save      - True save plot on pathOut, false none
    y1        - first signal  (mlp)
    y2        - second signal (of)
    y3        - third signal (target)
    
    """
    plt.rcParams["figure.figsize"] = (10,6)
    adjustXlim = kwargs.get('adjustXlim')
    titleName  = kwargs.get('titleName')
    MeanRms    = kwargs.get('MeanRms')
    fileName   = kwargs.get('fileName')
    pathOut    = kwargs.get('pathOut')
    label      = kwargs.get('label')
    legend     = kwargs.get('legend')
    xscale     = kwargs.get('xscale')
    log        = kwargs.get('log')
    show       = kwargs.get('show')
    save       = kwargs.get('save')
    ext        = kwargs.get('ext')
    detail     = kwargs.get('detail')
    unit       = kwargs.get('unit')
        
    if unit is None: unit = 'GeV'
    if pathOut is None: 
        pathOut = ''
        save = False
        show = True
    else :
        if show    is None: show = False
        if save    is None: save = False
    if label   is None: label = ''
    if legend  is None: legend = ['','','']
    if detail  is None: detail = False    
    if log     is None: log  = False
    if ext     is None: ext  = 'pdf'    
    if MeanRms  is None: MeanRms = 'mean'    
    if titleName is None: titleName = ''
    if fileName is None: fileName = ''
    if adjustXlim is None: adjustXlim = False
    scale_x = 1    
    
    if (label.lower() == 'energy'):
        if y2 is not None and y3 is not None: 
            if rms(y1) < 0.01 or rms(y2) < 0.01 or rms(y3) < 0.01:
                y1, y2, y3 = y1*1e3, y2*1e3, y3*1e3
                xLabel = r'Energy [$10^{-3}$'+f' {unit}]'
                #unit   = 'MeV'
            else :
                xLabel = f'Energy [{unit}]'
                #unit   = 'GeV'
        elif y2 is not None:
            if rms(y1) < 0.01 or rms(y2) < 0.01:
                y1, y2 = y1*1e3, y2*1e3                
                xLabel = r'Energy [$10^{-3}$'+f' {unit}]'
                #unit   = 'MeV'
            else :
                xLabel = f'Energy [{unit}]'
                #unit   = 'GeV'
        else:
            if rms(y1) < 0.01:
                y1 = y1*1e3
                xLabel = r'Energy [$10^{-3}$'+f' {unit}]'
                #unit   = 'MeV'
            else :
                xLabel = f'Energy [{unit}]'
                #unit   = 'GeV'
    elif label.lower() == 'time':
        xLabel = 'Time [ns]'
        unit   = 'ns'
    elif label.lower() == 'ss':
            xscale = 1
            xLabel = label
            unit   = ''        
    else :
        if rms(y1) < 0.01 or rms(y2) < 0.01 or rms(y3) < 0.01:
            #if xscale  is None: 
            #xscale  = 1e3
            xscale  = 1
            y1, y2, y3 = y1*xscale, y2*xscale, y3*xscale
            #xLabel = f'{label} '+r'[$10^{-3}$]'
            xLabel = f'{label}'
        else: 
            xscale = 1
            xLabel = label
            unit   = ''                
        
    Nbins, Ndpi  = 100, 700    
    minBin, maxBin = min_max(y1, y2, y3)
    valueBins = np.linspace(minBin, maxBin, Nbins)
    
    if detail is True:
        fig, ax = plt.subplots(1, 2, gridspec_kw={'width_ratios': [10, 6]})
        maxStd = maxBin
        meanMean = np.mean([y1, y2, y3])
        y1MeanRms, y2MeanRms, y3MeanRms = getMeanRms(y1, x2=y2, x3=y3, MeanRms=MeanRms)
        ## ==========================
        ## ==============
        ## THREE plots
        #n, bins, patches = ax.hist(y1,  bins=valueBins,  ec='navy',       alpha=0.7, fc='royalblue', lw=1.2, histtype='stepfilled', label=f'{legend[0]}'.ljust(14,' ')+f'rms: {rms(y1):.2f} $\pm$ {rmsErr(y1):.2f} {unit}'.ljust(28,' '))
        n, bins, patches = ax[0].hist(y1,  bins=valueBins,  ec='navy', alpha=0.7, fc='royalblue', lw=1.2, histtype='stepfilled', label=f'{legend[0]}'.ljust(14,' ')+ f': {y1MeanRms[0]:.2f} $\pm$ {y1MeanRms[1]:.2f} '.ljust(20,' '))
        ax[0].hist(y2,  bins=valueBins,  ec='red',  alpha=0.5, fc='indianred', lw=1.2, histtype='stepfilled', label=f'{legend[1]}'.ljust(14,' ')+ f': {y2MeanRms[0]:.2f} $\pm$ {y2MeanRms[1]:.2f} '.ljust(20,' '))
        ax[0].hist(y3,  bins=valueBins,  ec='k',    alpha=0.4, fc='gainsboro', lw=1.8, histtype='stepfilled', label=f'{legend[2]}'.ljust(14,' ')+ f': {y3MeanRms[0]:.2f} $\pm$ {y3MeanRms[1]:.2f} '.ljust(20,' '))
    
        xmin,xmax,ymin,ymax = ax[0].axis()
        ax[0].legend(frameon=False)
        #nn = np.where(n>10)
        
        if log == True:
            ax[0].set_yscale('log')
            ax[0].set_ylim([10, 20*abs(ymax)])
        elif len(y1) > 5e3:
            scale_y = 1e3
            ax[0].set_ylim([None, 1.4*ymax])
            ax[0].set_ylabel(f'Count'+r' [10$^3$]')    
            
            ticks_y = ticker.FuncFormatter(lambda y1, pos: '{0:g}'.format(y1/scale_y))
            ax[0].yaxis.set_major_formatter(ticks_y)
        else: ax[0].set_ylabel('Count')            
            
        ax[0].set_xlabel(f'{xLabel}')         
        ax[0].grid(ls='--', lw=0.7)
        ax[0].spines['top'].set_visible(False)
        ax[0].spines['right'].set_visible(False)
        
        if adjustXlim is True:
            if len(nn[0]) < 40:
                ii = list(n).index(np.max(n))
                if ii+5 > 99:
                    ax[0].set_xlim([bins[ii-5], bins[-1]])
                elif ii-5 < 0:
                    ax[0].set_xlim([bins[0], bins[ii+5]])
                else :
                    ax[0].set_xlim([bins[ii-5], bins[ii+5]])
        
        #minBin = min(min(y1), min(y3))
        #maxBin = max(max(y1), max(y3))
        #valueBins = np.linspace(minBin, maxBin, Nbins)

        n, bins, patches = ax[1].hist(y1,  bins=valueBins,  ec='navy', alpha=0.7, fc='royalblue', lw=1.2, histtype='stepfilled', label=f'{legend[0]}')
        ax[1].hist(y3,  bins=valueBins,  ec='k',    alpha=0.5, fc='gainsboro', lw=1.6, histtype='stepfilled', label=f'{legend[2]}')
        xmin,xmax,ymin,ymax = ax[1].axis()
        
        nn = np.where(n>10)
        if adjustXlim is True:
            if len(nn[0]) < 40:
                ii = list(n).index(np.max(n))
                if ii+5 > 99:
                    ax[1].set_xlim([bins[ii-5], bins[-1]])
                elif ii-5 < 0:
                    ax[1].set_xlim([bins[0], bins[ii+5]])
                else :
                    ax[1].set_xlim([bins[ii-5], bins[ii+5]])
        
        ax[1].legend(frameon=False)
        ax[1].yaxis.set_major_formatter(plt.NullFormatter())
        if log == True:
            ax[1].set_yscale('log')
            ax[1].set_ylim([10, 20*abs(ymax)])
        elif len(y1) > 5e3:
            scale_y = 1e3
            ax[1].set_ylim([None, 1.4*ymax])
            ticks_y = ticker.FuncFormatter(lambda y1, pos: '{0:g}'.format(y1/scale_y))
            ax[1].yaxis.set_major_formatter(ticks_y)
            
        #ax[1].axes.yaxis.set_visible(False)
        ax[1].grid(ls='--', lw=0.7)
        ax[1].set_xlabel(f'{xLabel}') 
        ax[1].spines['top'].set_visible(False)
        ax[1].spines['right'].set_visible(False)        
                
    else :        
        fig    = plt.figure()
        ax     = fig.add_subplot(111)
        if (y2 is not None and y3 is not None):
            y1MeanRms, y2MeanRms, y3MeanRms = getMeanRms(y1, x2=y2, x3=y3, MeanRms=MeanRms)
            maxStd = max(np.std(y1), np.std(y2), np.std(y3))
            meanMean = np.mean([y1, y2, y3])            
            
            n, bins, patches = ax.hist(y1,  bins=valueBins,  ec='navy',       alpha=0.7, fc='royalblue', lw=1.2, histtype='stepfilled', label=f'{legend[0]}'.ljust(14,' ')+ f'{MeanRms}: {y1MeanRms[0]:.2f} $\pm$ {y1MeanRms[1]:.2f} {unit}'.ljust(28,' '))
            #ax.hist(y1,  bins=valueBins,  ec='navy', alpha=0.7, fc='royalblue', lw=1.2, histtype='stepfilled', label=f'{legend[0]}'.ljust(14,' ')+ f'{MeanRms}: {y1MeanRms:.2f} $\pm$ {y1StdRmse:.2f} {unit}'.ljust(28,' '))
            ax.hist(y2,  bins=valueBins,  ec='red',  alpha=0.6, fc='indianred', lw=1.2, histtype='stepfilled', label=f'{legend[1]}'.ljust(14,' ')+ f'{MeanRms}: {y2MeanRms[0]:.2f} $\pm$ {y2MeanRms[1]:.2f} {unit}'.ljust(28,' '))            
            ax.hist(y3,  bins=valueBins,  ec='k',    alpha=0.5, fc='gainsboro', lw=1.5, histtype='stepfilled', label=f'{legend[2]}'.ljust(14,' ')+ f'{MeanRms}: {y3MeanRms[0]:.2f} $\pm$ {y3MeanRms[1]:.2f} {unit}'.ljust(28,' '))

            nn = np.where(n>10)
                ## --------------- ##
        ## ==========================
        ## ==============
        ## TWO plots
        elif (y2 is not None):
            y1MeanRms, y2MeanRms = getMeanRms(y1, x2=y2, MeanRms=MeanRms)
            maxStd = max(np.std(y1), np.std(y2))
            meanMean = np.mean([y1, y2])        

            n, bins, patches = ax.hist(y1,  bins=valueBins,  ec='navy', alpha=0.7, fc='royalblue', lw=1.2, histtype='stepfilled', label=f'{legend[0]}'.ljust(14,' ')+ f'{MeanRms}: {y1MeanRms[0]:.2f} $\pm$ {y1MeanRms[1]:.2f} {unit}'.ljust(28,' '))            
            ax.hist(y2,  bins=valueBins,  ec='red',  alpha=0.6, fc='indianred', lw=1.2, histtype='stepfilled', label=f'{legend[1]}'.ljust(14,' ')+ f'{MeanRms}: {y2MeanRms[0]:.2f} $\pm$ {y2MeanRms[1]:.2f} {unit}'.ljust(28,' '))            

        ## ==========================
        ## ==============
        ## ONE plot
        else :
            y1MeanRms = getMeanRms(y1, MeanRms=MeanRms)
            maxStd = np.std(y1)
            meanMean = np.mean(y1)
            n, bins, patches = ax.hist(y1,  bins=valueBins,  ec='k',    alpha=0.5, fc='gainsboro', lw=1.7, histtype='stepfilled', label=f'{legend[0]}'.ljust(14,' ')+ f'{MeanRms}: {y1MeanRms[0]:.2f} $\pm$ {y1MeanRms[1]:.2f} {unit}'.ljust(28,' '))

        #ticks_x = ticker.FuncFormatter(lambda y1, pos: '{0:g}'.format(y1*scale_x))
        #ax.xaxis.set_major_formatter(ticks_x)

        ax.set_title(titleName)
        ax.set_xlabel(f'{xLabel}')
        ax.grid(linestyle='--',linewidth=.7)
        #ax.legend(frameon=False, loc=legend[3])
        ax.legend(frameon=False)

        xmin,xmax,ymin,ymax = ax.axis()
        nn = np.where(n>5)
        if log == True:
            plt.gca().set_yscale("log")
            #ax.set_ylim([None, 40*abs(ymax)])
            ax.set_ylim([100, 8*abs(ymax)])
            ax.set_ylabel(f'Count'+r' [10$^3$]')        
            scale_y = 1e3
            fileName = f'{fileName}_log'
        elif len(y1) > 5e3:
            scale_y = 1e3
            ax.set_ylim([None, 1.4*ymax])
            ax.set_ylabel(f'Count'+r' [10$^3$]')
        else :
            scale_y = 1
            ax.set_ylim([None, 1.4*ymax])
            ax.set_ylabel(f'Count')
            
        limit = 0.0
        if adjustXlim is True:
            if len(nn[0]) < 40:
                ii = list(n).index(np.max(n))
                if ii+5 > 99:
                    ax.set_xlim([bins[ii-5], bins[-1]])
                elif ii-5 < 0:
                    ax.set_xlim([bins[0], bins[ii+5]])
                else :
                    ax.set_xlim([bins[ii-5], bins[ii+5]])
            
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ticks_y = ticker.FuncFormatter(lambda y1, pos: '{0:g}'.format(y1/scale_y))
        ax.yaxis.set_major_formatter(ticks_y)
        ax.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.3f}"))

    if save == True:
        fig.savefig(f'{pathOut}/{fileName}.{ext}', format=ext, dpi=Ndpi)
    #print(f'Show plot: {show}')

    if show == False :
        plt.close(fig)
        plt.close()
    elif show == True:
        #plt.show()
        #plt.gca().set_yscale("log")
        plt.show()
    del fig  

### ============================= 
def plotHist2columns(legend, fileName, pathOut, signal, y1, y2, y3=None, diff=None, **kwargs):
    unit = kwargs.get('unit')
    ext  = kwargs.get('ext')
    
    if unit is None: unit = ' '
    if ext  is None: unit = 'pdf'
    ### for a detail on plot for time
    # y1 - Ypredict
    # y2 - Target values
    # y3 - Optimal Filter output
    y = copy.deepcopy(y3)
    if y3 == None:
        y3 = np.zeros(10)

    Ndpi = 500
    fig, ax = plt.subplots(1, 2, gridspec_kw={'width_ratios': [9, 6]})
    binMin = min(min(y1), min(y2), min(y3))
    binMax = max(max(y1), max(y2), max(y3))
    valueBins = np.linspace(binMin,binMax,100)

    ax[0].hist(y1,  bins=valueBins,  ec='navy', alpha=0.6, fc='royalblue', lw=1.2, histtype='stepfilled', label=f'{legend[0]} : {rms(y1):.2f} $\pm$ {rmsErr(y1):.2f}')
    ax[0].hist(y2,  bins=valueBins,  ec='red',  alpha=0.5, fc='indianred', lw=1.2, histtype='stepfilled', label=f'{legend[1]} : {rms(y2):.2f} $\pm$ {rmsErr(y2):.2f}')
    
    if y is not None:
        ax[0].hist(y3, bins=Nbins, ec='darkorange',     alpha=0.8, fc='none', lw=1.5, histtype='step', label=f'{legend[2]} : {rms(y3):.2f} $\pm$ {rmsErr(y3):.2f} {unit}')    
        
    xmin,xmax,ymin,ymax = ax[0].axis()
    ax[0].legend(frameon=False)
    ax[0].set_yscale('log')
    ax[0].set_ylim([10, 20*abs(ymax)])
    ax[0].set_xlabel(f'{signal} {unit}')
    ax[0].set_ylabel('count')
    
    ax[0].grid(ls='--', lw=0.7)

    if diff is True:
        binMin = min(y2-y1)
        binMax = max(y2-y1)
        valueBins = np.linspace(binMin,binMax,100)
        maxStd   = np.std(y2-y1)
        meanMean = np.mean(y2-y1)

        ax[1].hist(y2-y1,  bins=valueBins,  ec='k',     alpha=0.6, fc='gainsboro', lw=1.2, histtype='stepfilled', label=f'Diff : {rms(y2-y1):.2f} $\pm$ {rmsErr(y2-y1):.2f}')
        ax[1].legend(frameon=False)
        xmin,xmax,ymin,ymax = ax[1].axis()
        ax[1].yaxis.set_major_formatter(plt.NullFormatter())
        #ax[1].axes.yaxis.set_visible(False)
        ax[1].set_yscale('log')
        ax[1].set_ylim([10, 20*ymax])
        ax[1].grid(ls='--', lw=0.7)
        ax[1].set_xlabel(f'{signal} {unit}')
        ax[1].set_xlim([meanMean-6*maxStd, meanMean+6*maxStd])
    else:
        binMin = min(min(y1), min(y2))
        binMax = max(max(y1), max(y2))
        valueBins = np.linspace(binMin,binMax,100)

        ax[1].hist(y2, bins=valueBins, ec='darkorange', alpha=0.6, fc='none', lw=1.5, histtype='step', label=legend[1])
        ax[1].hist(y1, bins=valueBins, ec='navy',       alpha=0.6, fc='none', lw=1.5, histtype='step', label=legend[0])
        ax[1].legend(frameon=False)
        ax[1].yaxis.set_major_formatter(plt.NullFormatter())
        #ax[1].axes.yaxis.set_visible(False)
        ax[1].grid(ls='--', lw=0.7)
        ax[1].set_xlabel(f'{signal} {unit}') 
    ax[0].spines['top'].set_visible(False)
    ax[0].spines['right'].set_visible(False)
    ax[1].spines['top'].set_visible(False)
    ax[1].spines['right'].set_visible(False)
    #fig.savefig(f'{pathOut}/{fileName}.{ext}', format=ext, dpi=Ndpi)
    plt.show()
    plt.close(fig)
    
### ============================= 
def plotScatter(yPred, yTrue, **kwargs):
    #fig = plt.figure()    
    default_params = {
        "ext": "png",
        "fileName": "",
        "cell": "",
        "MeanRms": "mean",
        "pathOut": "",
        "prefix": " ",
        "save": False,        
        "show": True,
        "struct": "",
        "title": False,
        "text": [],
        "unit": " ",
        }
    params = {**default_params, **kwargs}
        
    Ndpi = 700
    m, b    = np.polyfit(yTrue, yPred, 1)
    fig, ax = plt.subplots(1, 2, gridspec_kw={'width_ratios': [10, 6]})
    plt.subplots_adjust(wspace=0.05) 
    X_plot  = np.linspace(yTrue.min(), yTrue.max(),len(yTrue))

    axes = plt.gca()
    #ax[0].scatter(yTrue, yPred, s=7, label=f'{params["struct"]} - Cell {params["cell"]}')
    ax[0].scatter(yTrue, yPred, s=7, label=f'cell: {params["cell"]}')
    ax[0].plot(X_plot, m*X_plot + b, '-',color='r', alpha=0.5)
    
    #ax[0].plot([min(yTest[:,idxCell]), max(yTest[:,idxCell])], [min(yPred[:,idxCell]), max(yPred[:,idxCell])],color='r',alpha=0.7)
    if params["title"] is True:
        ax[0].set_title(f'${{{params["prefix"]}}}_{{{params["struct"]}}}$ '+r'$\times$'+f' ${{{params["prefix"]}}}_{{true}}$')
    ax[0].grid(ls='--', lw=0.7)
    ax[0].legend(frameon=False)
    #ax[0].set_xlabel(f'Target {signal} {unit}')
    ax[0].set_xlabel(f'${{{params["prefix"]}}}_{{Ref}}$ {params["unit"]}')
    ax[0].set_ylabel(f'${{{params["prefix"]}}}_{{{params["struct"]}}}$ {params["unit"]}')
    ax[1].hist(yPred,bins=100, ec='navy',  alpha=0.6, fc='gainsboro', lw=1.5, histtype='stepfilled', orientation='horizontal')
    ax[1].yaxis.set_major_formatter(plt.NullFormatter())
    ax[1].set_title(f'${{{params["prefix"]}}}_{{{params["struct"]}}}$')
    #ax[1].legend(frameon=False)
    ax[1].set_xlabel('count')
    ax[1].grid(ls='--', lw=0.7)
    ax[1].set_xscale('log')
    if params["show"] is True:
        plt.show()
    
    ax[0].spines['top'].set_visible(False)
    ax[0].spines['right'].set_visible(False)
    ax[1].spines['top'].set_visible(False)
    ax[1].spines['right'].set_visible(False)

    if params["save"] is True:
        fig.savefig(f'{params["pathOut"]}/{params["fileName"]}.{params["ext"]}', format=f'{params["ext"]}', dpi=Ndpi)
    plt.close(fig)      
    del fig  


##===========================================================
##===========================================================
def plot_mse(history, fileName, labels, signal, **kwargs):
    log  = kwargs.get('log')
    n    = kwargs.get('n')
    ext  = kwargs.get('ext')
    Ndpi = kwargs.get('Ndpi')
    
    if log  is None: log=False
    if n    is None: n='-'
    if Ndpi is None: Ndpi=500
    if ext  is None: ext='pdf'
    fig = plt.figure()
    try:
        plt.plot(history.history['loss'], label=labels[0] )
        plt.plot(history.history['val_loss'], label=labels[1] )
    except :
        plt.plot(history['loss'], label=labels[0] )
        plt.plot(history['val_loss'], label=labels[1] )
    plt.title(f'Train error {signal} - nNeurons {n} ')
    plt.xlabel('epochs')
    plt.ylabel('MSE')
    #plt.yscale('log')
    plt.grid(linestyle='--',linewidth=.7)
    plt.legend(frameon=False)
    fig.savefig(f'{fileName}.{ext}', format=ext, dpi=Ndpi)
    plt.close(fig)
    
    
##===========================================================
### Plot Reconstruction distribution
def plot_RE_distribution(hList=None,model_type=None,ae=None,version=None,phase=None, metric=None,show_lines=False,bins=30,save_path=False,path=None,ext=None):
    #fig, axs = plt.subplots(nrows=1, ncols=3,figsize=(20,6))
    fig, axs = plt.subplots(nrows=2, ncols=5,figsize=(20,6))
    #fold=3
    ax = axs.ravel()
    for idx in range(nrows*ncols):
        fold=idx+1
        #fold_y = np.load(os.path.join(run_folder,'fold'+str(fold),'y_'+phase+'.npy'))
        fold_y = np.load(os.path.join(run_folder,f'fold{fold}y_{phase}.npy'))
        
        #re = np.load(os.path.join(run_folder,'fold'+str(fold),phase+'_'+metric+'.npy'))
        re = np.load(os.path.join(run_folder,f'fold{fold}{phase}_{metric}.npy'))
        re_mean = np.mean(re)
        re_std = np.std(re)
        p25 = np.percentile(re,25)
        p50 = np.median(re)
        p75 = np.percentile(re,75)
        xposition = [p25,p50,p75]
        #xposition = [re_mean-3*re_std, re_mean-2*re_std,re_mean+2*re_std,re_mean+3*re_std]

        re = np.load(os.path.join(run_folder,'fold'+str(fold),phase+'_'+metric+'.npy'))
        if bins is None:
            hist, bin_edges = np.histogram(re,range=(np.min(re),np.max(re)))
        else:
            hist, bin_edges = np.histogram(re,bins=bins,range=(np.min(re),np.max(re)))
        
        ax[idx].hist(np.load(os.path.join(run_folder,'fold'+str(fold),phase+'_'+metric+'.npy'))[np.where(fold_y == 0)],label='Normal', bins=bin_edges,alpha=0.5,color='tab:blue')
        ax[idx].hist(np.load(os.path.join(run_folder,'fold'+str(fold),phase+'_'+metric+'.npy'))[np.where(fold_y == 1)],label='TB', bins=bin_edges,alpha=0.5, color='tab:orange')
        ax[idx].set_xlabel("Reconstruction Error - "+metric)
        #ax[idx].xticks(rotation=45)
        ax[idx].tick_params(axis='x', rotation=45)
        ax[idx].legend()
        ax[idx].grid()
        ax[idx].set_title("Fold: "+str(fold)+" "+phase)
        
        if show_lines:
            for xc in xposition:
                ax[idx].axvline(x=xc, color='lightskyblue', linestyle='--')
            
    #fig.delaxes(ax[idx+1])
    #plt.suptitle(model_type+'-autoencoder-'+ae+'-V'+version,y=1.05)
    plt.suptitle(f'{model_type}-autoencoder-{ae}-V{version}',y=1.05)
    plt.tight_layout()
    if save_path:
        plt.savefig(path,format=ext, dpi=800)

## =========================================================
def plotClusters(TrainingSummary, version, run, pathAnalysis, pathDay, InputDim, pathOFlztEbins, **kwargs):
    
    model     = kwargs.get('model')
    signal    = kwargs.get('signal')
    energy    = kwargs.get('energy')
    save      = kwargs.get('save')
    show      = kwargs.get('show')
    typeTrain = kwargs.get('typeTrain')
    
    if save is None: save=False
    
    idx_7x7 = array(range(49))
    idx_5x5 = getIdxClus_mxn(idx_7x7.reshape(7,7), 5, 5)
    idx_3x3 = getIdxClus_mxn(idx_7x7.reshape(7,7), 3, 3)
    ij_cell = array(['-3,3' , '-2,3' , '-1,3' , '0,3' , '1,3' , '2,3' , '3,3' , 
                     '-3,2' , '-2,2' , '-1,2' , '0,2' , '1,2' , '2,2' , '3,2' , 
                     '-3,1' , '-2,1' , '-1,1' , '0,1' , '1,1' , '2,1' , '3,1' , 
                     '-3,0' , '-2,0' , '-1,0' , '0,0' , '1,0' , '2,0' , '3,0' , 
                     '-3,-1', '-2,-1', '-1,-1', '0,-1', '1,-1', '2,-1', '3,-1', 
                     '-3,-2', '-2,-2', '-1,-2', '0,-2', '1,-2', '2,-2', '3,-2', 
                     '-3,-3', '-2,-3', '-1,-3', '0,-3', '1,-3', '2,-3', '3,-3' ])

    ## -----------------------------------------------
    ## ---- get data: Target, OF, and Predicted
    dictParam = getDictParam(TrainingSummary, pathDay, typeTrain, pathOFlztEbins, model=model, energy=energy, save=save, signal=signal)
    ## -----------------------------------------------
    
    yTarAmp, yTarTime, yOFAmp, yOFTime, yPredAmp, yPredTime, order, nCell, _ = dictParam[f'{energy}'].values()
        
    foldKey  = list(TrainingSummary.keys())[0]
    try:
        model_type = TrainingSummary[foldKey]['ModelType']
        struct = TrainingSummary[foldKey]['Architecture']
        nNeuron = TrainingSummary[foldKey]['nNeuron']
        nLayers = TrainingSummary[foldKey]['HiddenLayer']
    except :
        model_type = TrainingSummary[foldKey]['info']['ModelType']
        struct = TrainingSummary[foldKey]['info']['Architecture']
        nNeuron = TrainingSummary[foldKey]['info']['nNeuron']
        nLayers = TrainingSummary[foldKey]['info']['HiddenLayer']
    
    rmse = np.sqrt(TrainingSummary[foldKey]['Eval']['Train']['BestEval'][0])
    inputDim = run[run.find('_In')+3:run.find('Lay')-2]

    pathAmp  = createPath(f'{pathAnalysis}/energy/mean')
    pathTime = createPath(f'{pathAnalysis}/time/mean')
    
    if  nCell    == len(idx_3x3):
        idxClus = idx_3x3        
        nCell = len(idx_3x3)
    elif  nCell  == len(idx_5x5):
        idxClus = idx_5x5
        nCell = len(idx_5x5)
    else: 
        idxClus = idx_7x7
        nCell = len(idx_7x7)
    
    if signal.lower() == 'energy':
        table3x3_E = []
        #fileTableE, table5x5_E = fileOpenClose(f'{pathAmp}/latexTableBest_{energy}_energy_train_{version}', 'open','w+')
        fileTableE, table5x5_E = fileOpenClose(f'{pathAnalysis}/latexTableBest_{energy}_energy_train_{version}', 'open','w+')
        table3x3_E, table5x5_E = latexTableAnalysis(table3x3_E, table5x5_E, run=run, rmse=rmse, yTarget=yTarAmp, yPred=yPredAmp, pos='begin', MeV_GeVns=order)        
    
        for i in range(nCell):
            fileName  = f'{model_type}_{struct}_energy_{nLayers}_Lay_{nNeuron}_Neu_In{InputDim}_Cell{idxClus[i]}_{version}_mean'
            titleName = f'Energy - Target '+r'$\times$'+f' E$_{{OF}}$ '+r'$\times$'+f' E$_{{{struct}}}$ | Cell {idxClus[i]}'
            legend = [r'$E_{mlp}$', r'$E_{of}$',r'$E_{true}$']

            plotHisto(yPredAmp[:,i], y2=yOFAmp[:,i], y3=yTarAmp[:,i], pathOut=pathAmp, label='energy', legend=legend, fileName=fileName, save=save, adjustXlim=True, detail=False, unit=order)

            errE_mlp = yPredAmp[:,i] - yTarAmp[:,i]
            errE_of  = yOFAmp[:,i] - yTarAmp[:,i]

            dictParam[energy].update({'Err_E_OF':errE_of, 'Err_E_MLP':errE_mlp})

            ## ----------------
            ## Error
            fileName  = f'error_energy_{struct}_OF_{nLayers}_Lay_{nNeuron}_Neu_In{InputDim}_Cell{idxClus[i]}_{version}_rms'
            plotHisto(errE_mlp, y2=errE_of, pathOut=pathAmp, label='Energy', legend=[f'Err$_{{{struct}}}$', r'Err$_{OF}$'], fileName=fileName, titleName=titleName, save=save, show=show, adjustXlim=False, MeanRms='rms', unit='MeV')

            ## ----------------
            ## Scatter
            fileName = f'scatter_mlp_x_target_energy_{nLayers}_Lay_{nNeuron}_Neu_In{InputDim}_Cell{idxClus[i]}_{version}'
            #plotScatter(pathAmp, yPredAmp[:, i], yTarAmp[:,i], idxClus[i], struct, fileName, save=True)

            fileName = f'scatter_mlp_x_target_time_{nLayers}_Lay_{nNeuron}_Neu_In{InputDim}_Cell{idxClus[i]}_{version}'            

            #table3x3_E, table5x5_E = latexTableAnalysis(table3x3_E, table5x5_E, yTarget=yTarAmp[:,i], OFxt=yOFAmp[:,i], yPred=yPredAmp[:,i], cell=ij_cell[i], pos='middle', idx=i)            
            table3x3_E, table5x5_E = latexTableAnalysis(table3x3_E, table5x5_E, yTarget=yTarAmp, OFxt=yOFAmp, yPred=yPredAmp, cell=ij_cell[i], pos='middle', idx=i)            

        table3x3_E, table5x5_E = latexTableAnalysis(table3x3_E, table5x5_E, yTarget=yTarAmp, OFxt=yOFAmp, yPred=yPredAmp, idx_3x3=idx_3x3, pos='end')        

        fileOpenClose(f'{pathAmp}/latexTableBest_energy_train_{version}', 'close','w+', fileTableE, table5x5_E)        
        
    elif signal.lower() == 'time':
        table3x3_T = []
        #fileTableT, table5x5_T = fileOpenClose(f'{pathTime}/latexTableBest_{energy}_time_train_{version}', 'open','w+')
        fileTableT, table5x5_T = fileOpenClose(f'{pathAnalysis}/latexTableBest_{energy}_time_train_{version}', 'open','w+')
        table3x3_T, table5x5_T = latexTableAnalysis(table3x3_T, table5x5_T, run=run, rmse=rmse, yTarget=yTarTime, yPred=yPredTime, pos='begin', MeV_GeVns='ns')
        for i in range(nCell):

            errT_mlp = yPredTime[:,i] - yTarTime[:,i]
            errT_of  = yOFTime[:,i] - yTarTime[:,i]

            dictParam[energy].update({'Err_T_OF':errT_of, 'Err_T_MLP':errT_of})

            ## ----------------
            ## Error
            fileName  = f'error_time_{struct}_OF_{nLayers}_Lay_{nNeuron}_Neu_In{InputDim}_Cell{idxClus[i]}_{version}_rms'
            #plotHisto(errT_mlp, y2=errT_of, pathOut=pathTime, label='Time', legend=[f'Err$_{{{struct}}}$', r'Err$_{OF}$'], fileName=fileName, titleName=titleName, save=True, adjustXlim=False, MeanRms='rms', unit='MeV')

            ## ----------------
            ## Scatter
            fileName  = f'{model_type}_{struct}_time_{nLayers}_Lay_{nNeuron}_Neu_In{InputDim}_Cell{idxClus[i]}_{version}_mean'
            titleName = r'$\tau$'+ f' - Target '+r'$\times$'+ r'$\tau_{{OF}}$ '+r'$\times$'+ r'$\tau_{{{struct}}}$ '+ f' | Cell {idxClus[i]}'
            legend = [r'$\tau_{mlp}$', r'$\tau_{of}$',r'$\tau_{true}$']

            #plotHisto(yPredTime[:,i], y2=yOFTime[:,i], y3=yTarTime[:,i], pathOut=pathTime, label='time', legend=legend, fileName=fileName, save=True, adjustXlim=True, detail=False)

            fileName = f'scatter_mlp_x_target_time_{nLayers}_Lay_{nNeuron}_Neu_In{InputDim}_Cell{idxClus[i]}_{version}'                        
            #print(f'{i}')
         #   #plotScatter(pathTime, yPredTime[:, i], yTarTime[:,i], idxClus[i], struct, fileName, save=save)        

            #table3x3_T, table5x5_T = latexTableAnalysis(table3x3_T, table5x5_T, yTarget=yTarTime[:,i], OFxt=yOFTime[:,i], yPred=yPredTime[:,i], cell=ij_cell[i], pos='middle', MeV_GeVns='ns', idx=i)
            
            table3x3_T, table5x5_T = latexTableAnalysis(table3x3_T, table5x5_T, yTarget=yTarTime, OFxt=yOFTime, yPred=yPredTime, cell=ij_cell[i], pos='middle', MeV_GeVns='ns', idx=i)
        
        table3x3_T, table5x5_T = latexTableAnalysis(table3x3_T, table5x5_T, yTarget=yTarTime, OFxt=yOFTime, yPred=yPredTime, idx_3x3=idx_3x3, pos='end')
        
        fileOpenClose(f'{pathTime}/latexTableBest_time_train_{version}', 'close','w+', fileTableT, table5x5_T)

    else:
        table3x3_T,table3x3_E = [], []
        fileTableE, table5x5_E = fileOpenClose(f'{pathAnalysis}/latexTableBest_{energy}_energy_train_{version}', 'open','w+')
        table3x3_E, table5x5_E = latexTableAnalysis(table3x3_E, table5x5_E, run=run, rmse=rmse, yTarget=yTarAmp, yPred=yPredAmp, pos='begin', MeV_GeVns=order)        

        fileTableT, table5x5_T = fileOpenClose(f'{pathAnalysis}/latexTableBest_{energy}_time_train_{version}', 'open','w+')
        table3x3_T, table5x5_T = latexTableAnalysis(table3x3_T, table5x5_T, run=run, rmse=rmse, yTarget=yTarTime, yPred=yPredTime, pos='begin', MeV_GeVns='ns')
        for i in range(nCell):
            fileName  = f'{model_type}_{struct}_energy_{nLayers}_Lay_{nNeuron}_Neu_In{InputDim}_Cell{idxClus[i]}_{version}_mean'
            titleName = f'Energy - Target '+r'$\times$'+f' E$_{{OF}}$ '+r'$\times$'+f' E$_{{{struct}}}$ | Cell {idxClus[i]}'
            legend = [r'$E_{mlp}$', r'$E_{of}$',r'$E_{true}$']

            plotHisto(yPredAmp[:,i], y2=yOFAmp[:,i], y3=yTarAmp[:,i], pathOut=pathAmp, label='energy', legend=legend, fileName=fileName, save=save, show=show, adjustXlim=True, detail=False, unit=order)

            errE_mlp = yPredAmp[:,i] - yTarAmp[:,i]
            errE_of  = yOFAmp[:,i] - yTarAmp[:,i]

            errT_mlp = yPredTime[:,i] - yTarTime[:,i]
            errT_of  = yOFTime[:,i] - yTarTime[:,i]

            dictParam[energy].update({'Err_E_OF':errE_of, 'Err_E_MLP':errE_mlp, 'Err_T_OF':errT_of, 'Err_T_MLP':errT_of})

            ## ----------------
            ## Error
            fileName  = f'error_energy_{struct}_OF_{nLayers}_Lay_{nNeuron}_Neu_In{InputDim}_Cell{idxClus[i]}_{version}_rms'
            plotHisto(errE_mlp, y2=errE_of, pathOut=pathAmp, label='Energy', legend=[f'Err$_{{{struct}}}$', r'Err$_{OF}$'], fileName=fileName, titleName=titleName, save=save, show=show, adjustXlim=False, MeanRms='rms', unit='MeV')

            fileName  = f'error_time_{struct}_OF_{nLayers}_Lay_{nNeuron}_Neu_In{InputDim}_Cell{idxClus[i]}_{version}_rms'
            #plotHisto(errT_mlp, y2=errT_of, pathOut=pathTime, label='Time', legend=[f'Err$_{{{struct}}}$', r'Err$_{OF}$'], fileName=fileName, titleName=titleName, save=True, adjustXlim=False, MeanRms='rms', unit='MeV')

            ## ----------------
            ## Scatter
            fileName = f'scatter_mlp_x_target_energy_{nLayers}_Lay_{nNeuron}_Neu_In{InputDim}_Cell{idxClus[i]}_{version}'
            #plotScatter(pathAmp, yPredAmp[:, i], yTarAmp[:,i], idxClus[i], struct, fileName, save=True)

            fileName  = f'{model_type}_{struct}_time_{nLayers}_Lay_{nNeuron}_Neu_In{InputDim}_Cell{idxClus[i]}_{version}_mean'
            titleName = r'$\tau$'+ f' - Target '+r'$\times$'+ r'$\tau_{{OF}}$ '+r'$\times$'+ r'$\tau_{{{struct}}}$ '+ f' | Cell {idxClus[i]}'
            legend = [r'$\tau_{mlp}$', r'$\tau_{of}$',r'$\tau_{true}$']

            #plotHisto(yPredTime[:,i], y2=yOFTime[:,i], y3=yTarTime[:,i], pathOut=pathTime, label='time', legend=legend, fileName=fileName, save=True, adjustXlim=True, detail=False)

            fileName = f'scatter_mlp_x_target_time_{nLayers}_Lay_{nNeuron}_Neu_In{InputDim}_Cell{idxClus[i]}_{version}'            
            if yTarTime != 0:
                plotScatter(pathTime, yPredTime[:, i], yTarTime[:,i], idxClus[i], struct, fileName, save=save)        

            #table3x3_E, table5x5_E = latexTableAnalysis(table3x3_E, table5x5_E, yTarget=yTarAmp[:,i], OFxt=yOFAmp[:,i], yPred=yPredAmp[:,i], cell=ij_cell[i], pos='middle', idx=i)
            #table3x3_T, table5x5_T = latexTableAnalysis(table3x3_T, table5x5_T, yTarget=yTarTime[:,i], OFxt=yOFTime[:,i], yPred=yPredTime[:,i], cell=ij_cell[i], pos='middle', MeV_GeVns='ns', idx=i)

            table3x3_E, table5x5_E = latexTableAnalysis(table3x3_E, table5x5_E, yTarget=yTarAmp, OFxt=yOFAmp, yPred=yPredAmp, cell=ij_cell[i], pos='middle', idx=i)
            table3x3_T, table5x5_T = latexTableAnalysis(table3x3_T, table5x5_T, yTarget=yTarTime, OFxt=yOFTime, yPred=yPredTime, cell=ij_cell[i], pos='middle', MeV_GeVns='ns', idx=i)


        table3x3_E, table5x5_E = latexTableAnalysis(table3x3_E, table5x5_E, yTarget=yTarAmp, OFxt=yOFAmp, yPred=yPredAmp, idx_3x3=idx_3x3, pos='end')
        table3x3_T, table5x5_T = latexTableAnalysis(table3x3_T, table5x5_T, yTarget=yTarTime, OFxt=yOFTime, yPred=yPredTime, idx_3x3=idx_3x3, pos='end')

        fileOpenClose(f'{pathAmp}/latexTableBest_energy_train_{version}', 'close','w+', fileTableE, table5x5_E)
        fileOpenClose(f'{pathTime}/latexTableBest_time_train_{version}', 'close','w+', fileTableT, table5x5_T)

    #return table5x5_E
    return dictParam

## =========================================================
def _plotClusters(TrainingSummary, version, run, pathAnalysis, pathDay, InputDim, typeTrain, energy=None, save=False):
    ij_5x5 = ['-2,2',  '-1,2',  '0,2',  '1,2',  '2,2',  '-2,1',  '-1,1',  '0,1',  '1,1',  '2,1', '-2,0',  '-1,0',  '0,0',  '1,0',  '2,0', '-2,-1', '-1,-1', '0,-1', '1,-1', '2,-1', '-2,-2', '-1,-2', '0,-2', '1,-2', '2,-2']
    idx_3x3 = ['-1,1',  '0,1',  '1,1',  '-1,0',  '0,0',  '1,0',  '-1,-1', '0,-1', '1,-1' ]
    
    #ij_5x5 = ['00', '01', '02', '03', '04', '10', '11', '12', '13', '14', '20', '21', '22', '23', '24', '30', '31', '32', '33', '34', '40', '41', '42', '43', '44']
    #idx_3x3 = ['11', '12', '13', '21', '22', '23', '31', '32', '33']
    
    idx_3x3 = [6, 7, 8, 11, 12, 13, 16, 17, 18]
    #yTarAmp, yTarTime, yOFAmp, yOFTime, yPredAmp, yPredTime, order = getDictParam(TrainingSummary, pathDay, energy, save)
    dictParam = getDictParam(TrainingSummary, pathDay, energy, save)
    
    yTarAmp, yTarTime, yOFAmp, yOFTime, yPredAmp, yPredTime, order, _ = dictParam[f'{energy}'].values()
    
    nCell = yTarAmp.shape[1]
        
    foldKey  = list(TrainingSummary.keys())[0]
    try:
        model_type = TrainingSummary[foldKey]['ModelType']
        struct = TrainingSummary[foldKey]['Architecture']
        nNeuron = TrainingSummary[foldKey]['nNeuron']
        nLayers = TrainingSummary[foldKey]['HiddenLayer']
    except :
        model_type = TrainingSummary[foldKey]['info']['ModelType']
        struct = TrainingSummary[foldKey]['info']['Architecture']
        nNeuron = TrainingSummary[foldKey]['info']['nNeuron']
        nLayers = TrainingSummary[foldKey]['info']['HiddenLayer']
    
    rmse = np.sqrt(TrainingSummary[foldKey]['Eval']['Train']['BestEval'][0])
    inputDim = run[run.find('_In')+3:run.find('Lay')-2]
    #energy = run[run.find('_e')+1:run.find('GeV')+3]
    
    #if energy is None:
        
    #pathAmp  = createPath(f'{pathDay}/analysis/{energy}/energy/mean')
    #pathTime = createPath(f'{pathDay}/analysis/{energy}/time/mean')
    pathAmp  = createPath(f'{pathAnalysis}/energy/mean')
    pathTime = createPath(f'{pathAnalysis}/time/mean')

    fileTableT, table5x5_T = fileOpenClose(f'{pathTime}/latexTableBest_{energy}_time_train_{version}', 'open','w+')
    fileTableE, table5x5_E = fileOpenClose(f'{pathAmp}/latexTableBest_{energy}_energy_train_{version}', 'open','w+')
    
    table3x3_E, table3x3_T = [], []
    
    if nCell == 9: ij = idx_3x3
    else : ij = ij_5x5
    
    table3x3_E, table5x5_E = latexTableAnalysis(table3x3_E, table5x5_E, run=run, rmse=rmse, yTarget=yTarAmp, yPred=yPredAmp, pos='begin', MeV_GeVns=order)
    table3x3_T, table5x5_T = latexTableAnalysis(table3x3_T, table5x5_T, run=run, rmse=rmse, yTarget=yTarTime, yPred=yPredTime, pos='begin', MeV_GeVns='ns')
    
    for i in range(nCell):
        fileName  = f'{model_type}_{struct}_energy_{nLayers}_Lay_{nNeuron}_Neu_In{InputDim}_Cell{ij[i]}_{version}_mean'
        titleName = f'Energy - Target '+r'$\times$'+f' E$_{{OF}}$ '+r'$\times$'+f' E$_{{{struct}}}$ | Cell {ij[i]}'
        legend = [r'$E_{mlp}$', r'$E_{of}$',r'$E_{true}$']
        
        plotHisto(yPredAmp[:,i], y2=yOFAmp[:,i], y3=yTarAmp[:,i], pathOut=pathAmp, label='energy', legend=legend, fileName=fileName, save=save, adjustXlim=True, detail=False, unit=order)

        errE_mlp = yPredAmp[:,i] - yTarAmp[:,i]
        errE_of  = yOFAmp[:,i] - yTarAmp[:,i]
        
        errT_mlp = yPredTime[:,i] - yTarTime[:,i]
        errT_of  = yOFTime[:,i] - yTarTime[:,i]
        
        dictParam[energy].update({'Err_E_OF':errE_of, 'Err_E_MLP':errE_mlp, 'Err_T_OF':errT_of, 'Err_T_MLP':errT_of})
        
        ## Error
        fileName  = f'error_energy_{struct}_OF_{nLayers}_Lay_{nNeuron}_Neu_In{InputDim}_Cell{ij[i]}_{version}_rms'
        plotHisto(errE_mlp, y2=errE_of, pathOut=pathAmp, label='Energy', legend=[f'Err$_{{{struct}}}$', r'Err$_{OF}$'], fileName=fileName, titleName=titleName, save=save, adjustXlim=False, MeanRms='rms', unit='MeV')
            
        fileName  = f'error_time_{struct}_OF_{nLayers}_Lay_{nNeuron}_Neu_In{InputDim}_Cell{ij[i]}_{version}_rms'
        #plotHisto(errT_mlp, y2=errT_of, pathOut=pathTime, label='Time', legend=[f'Err$_{{{struct}}}$', r'Err$_{OF}$'], fileName=fileName, titleName=titleName, save=True, adjustXlim=False, MeanRms='rms', unit='MeV')
        
        ## Scatter
        fileName = f'scatter_mlp_x_target_energy_{nLayers}_Lay_{nNeuron}_Neu_In{InputDim}_Cell{ij[i]}_{version}'
        #plotScatter(pathAmp, yPredAmp[:, i], yTarAmp[:,i], ij[i], struct, fileName, save=True)
            
        fileName  = f'{model_type}_{struct}_time_{nLayers}_Lay_{nNeuron}_Neu_In{InputDim}_Cell{ij[i]}_{version}_mean'
        titleName = r'$\tau$'+ f' - Target '+r'$\times$'+ r'$\tau_{{OF}}$ '+r'$\times$'+ r'$\tau_{{{struct}}}$ '+ f' | Cell {ij[i]}'
        legend = [r'$\tau_{mlp}$', r'$\tau_{of}$',r'$\tau_{true}$']
        
        #plotHisto(yPredTime[:,i], y2=yOFTime[:,i], y3=yTarTime[:,i], pathOut=pathTime, label='time', legend=legend, fileName=fileName, save=True, adjustXlim=True, detail=False)

        fileName = f'scatter_mlp_x_target_time_{nLayers}_Lay_{nNeuron}_Neu_In{InputDim}_Cell{ij[i]}_{version}'            
        plotScatter(pathTime, yPredTime[:, i], yTarTime[:,i], ij[i], struct, fileName, save=save)        
        
        table3x3_E, table5x5_E = latexTableAnalysis(table3x3_E, table5x5_E, yTarget=yTarAmp[:,i], OFxt=yOFAmp[:,i], yPred=yPredAmp[:,i], cell=ij_5x5[i], pos='middle', idx=i)
        table3x3_T, table5x5_T = latexTableAnalysis(table3x3_T, table5x5_T, yTarget=yTarTime[:,i], OFxt=yOFTime[:,i], yPred=yPredTime[:,i], cell=ij_5x5[i], pos='middle', MeV_GeVns='ns', idx=i)

    table3x3_E, table5x5_E = latexTableAnalysis(table3x3_E, table5x5_E, yTarget=yTarAmp, OFxt=yOFAmp, yPred=yPredAmp, idx_3x3=idx_3x3, pos='end')
    table3x3_T, table5x5_T = latexTableAnalysis(table3x3_T, table5x5_T, yTarget=yTarTime, OFxt=yOFTime, yPred=yPredTime, idx_3x3=idx_3x3, pos='end')
    
    fileOpenClose(f'{pathAmp}/latexTableBest_energy_train_{version}', 'close','w+', fileTableE, table5x5_E)
    fileOpenClose(f'{pathTime}/latexTableBest_time_train_{version}', 'close','w+', fileTableT, table5x5_T)
    
    #return table5x5_E
    return dictParam

##================

def __plotClusters(TrainingSummary, version, run, pathAnalysis, pathDay, energy=None, save=False):
    ij_5x5 = ['-2,2',  '-1,2',  '0,2',  '1,2',  '2,2',  '-2,1',  '-1,1',  '0,1',  '1,1',  '2,1', '-2,0',  '-1,0',  '0,0',  '1,0',  '2,0', '-2,-1', '-1,-1', '0,-1', '1,-1', '2,-1', '-2,-2', '-1,-2', '0,-2', '1,-2', '2,-2']
    idx_3x3 = ['-1,1',  '0,1',  '1,1',  '-1,0',  '0,0',  '1,0',  '-1,-1', '0,-1', '1,-1', ]
    
    #ij_5x5 = ['00', '01', '02', '03', '04', '10', '11', '12', '13', '14', '20', '21', '22', '23', '24', '30', '31', '32', '33', '34', '40', '41', '42', '43', '44']
    #idx_3x3 = ['11', '12', '13', '21', '22', '23', '31', '32', '33']
    
    idx_3x3 = [6, 7, 8, 11, 12, 13, 16, 17, 18]
    yTarAmp, yTarTime, yOFAmp, yOFTime, yPredAmp, yPredTime, order = getDictParam(TrainingSummary, pathDay, energy, save)
    nCell = yTarAmp.shape[1]
        
    foldKey  = list(TrainingSummary.keys())[0]
    try:
        model_type = TrainingSummary[foldKey]['ModelType']
        struct = TrainingSummary[foldKey]['Architecture']
        nNeuron = TrainingSummary[foldKey]['nNeuron']
        nLayers = TrainingSummary[foldKey]['HiddenLayer']
    except :
        model_type = TrainingSummary[foldKey]['info']['ModelType']
        struct = TrainingSummary[foldKey]['info']['Architecture']
        nNeuron = TrainingSummary[foldKey]['info']['nNeuron']
        nLayers = TrainingSummary[foldKey]['info']['HiddenLayer']
    
    rmse = np.sqrt(TrainingSummary[foldKey]['Eval']['Train']['BestEval'][0])
    InputDim = run[run.find('_In')+3:run.find('Lay')-2]
    #energy = run[run.find('_e')+1:run.find('GeV')+3]
    
    #if energy is None:
        
    #pathAmp  = createPath(f'{pathDay}/analysis/{energy}/energy/mean')
    #pathTime = createPath(f'{pathDay}/analysis/{energy}/time/mean')
    pathAmp  = createPath(f'{pathAnalysis}/energy/mean')
    pathTime = createPath(f'{pathAnalysis}/time/mean')

    fileTableT, table5x5_T = fileOpenClose(f'{pathTime}/latexTableBest_{energy}_{nNeuron}_neu_time_train_{version}', 'open','w+')
    fileTableE, table5x5_E = fileOpenClose(f'{pathAmp}/latexTableBest_{energy}_{nNeuron}_neu_energy_train_{version}', 'open','w+')
    
    table3x3_E, table3x3_T = [], []
    
    if nCell == 9: ij = idx_3x3
    else : ij = ij_5x5
    
    table3x3_E, table5x5_E = latexTableAnalysis(table3x3_E, table5x5_E, run=run, rmse=rmse, yTarget=yTarAmp, yPred=yPredAmp, pos='begin', MeV_GeVns=order)
    table3x3_T, table5x5_T = latexTableAnalysis(table3x3_T, table5x5_T, run=run, rmse=rmse, yTarget=yTarTime, yPred=yPredTime, pos='begin', MeV_GeVns='ns')
    
    for i in range(nCell):
        fileName  = f'{model_type}_{struct}_energy_{nLayers}_Lay_{nNeuron}_Neu_In{InputDim}_Cell{ij[i]}_{version}_mean'
        titleName = f'Energy - Target '+r'$\times$'+f' E$_{{OF}}$ '+r'$\times$'+f' E$_{{{struct}}}$ | Cell {ij[i]}'
        legend = [r'$E_{mlp}$', r'$E_{of}$',r'$E_{true}$']
        
        plotHisto(yPredAmp[:,i], y2=yOFAmp[:,i], y3=yTarAmp[:,i], pathOut=pathAmp, label='energy', legend=legend, fileName=fileName, save=True, adjustXlim=True, detail=False, unit=order)

        errE_mlp = yPredAmp[:,i] - yTarAmp[:,i]
        errE_of  = yOFAmp[:,i] - yTarAmp[:,i]
        
        errT_mlp = yPredTime[:,i] - yTarTime[:,i]
        errT_of  = yOFTime[:,i] - yTarTime[:,i]
        
        ## Error
        fileName  = f'error_energy_{struct}_OF_{nLayers}_Lay_{nNeuron}_Neu_In{InputDim}_Cell{ij[i]}_{version}_rms'
        plotHisto(errE_mlp, y2=errE_of, pathOut=pathAmp, label='Energy', legend=[f'Err$_{{{struct}}}$', r'Err$_{OF}$'], fileName=fileName, titleName=titleName, save=save, adjustXlim=False, MeanRms='rms', unit=order)
            
        fileName  = f'error_time_{struct}_OF_{nLayers}_Lay_{nNeuron}_Neu_In{InputDim}_Cell{ij[i]}_{version}_rms'
        plotHisto(errT_mlp, y2=errT_of, pathOut=pathTime, label='Time', legend=[f'Err$_{{{struct}}}$', r'Err$_{OF}$'], fileName=fileName, titleName=titleName, save=save, adjustXlim=False, MeanRms='rms')
        
        ## Scatter
        fileName = f'scatter_mlp_x_target_energy_{nLayers}_Lay_{nNeuron}_Neu_In{InputDim}_Cell{ij[i]}_{version}'
        plotScatter(pathAmp, yPredAmp[:, i], yTarAmp[:,i], ij[i], struct, fileName, save=save)
            
        fileName  = f'{model_type}_{struct}_time_{nLayers}_Lay_{nNeuron}_Neu_In{InputDim}_Cell{ij[i]}_{version}_mean'
        titleName = r'$\tau$'+ f' - Target '+r'$\times$'+ r'$\tau_{{OF}}$ '+r'$\times$'+ r'$\tau_{{{struct}}}$ '+ f' | Cell {ij[i]}'
        legend = [r'$\tau_{mlp}$', r'$\tau_{of}$',r'$\tau_{true}$']
        
        plotHisto(yPredTime[:,i], y2=yOFTime[:,i], y3=yTarTime[:,i], pathOut=pathTime, label='time', legend=legend, fileName=fileName, save=save, adjustXlim=True, detail=False)

        fileName = f'scatter_mlp_x_target_time_{nLayers}_Lay_{nNeuron}_Neu_In{InputDim}_Cell{ij[i]}_{version}'            
        plotScatter(pathTime, yPredTime[:, i], yTarTime[:,i], ij[i], struct, fileName, save=save)        
        
        table3x3_E, table5x5_E = latexTableAnalysis(table3x3_E, table5x5_E, yTarget=yTarAmp[:,i], OFxt=yOFAmp[:,i], yPred=yPredAmp[:,i], cell=ij_5x5[i], pos='middle', idx=i)
        table3x3_T, table5x5_T = latexTableAnalysis(table3x3_T, table5x5_T, yTarget=yTarTime[:,i], OFxt=yOFTime[:,i], yPred=yPredTime[:,i], cell=ij_5x5[i], pos='middle', MeV_GeVns='ns', idx=i)

    table3x3_E, table5x5_E = latexTableAnalysis(table3x3_E, table5x5_E, yTarget=yTarAmp, OFxt=yOFAmp, yPred=yPredAmp, idx_3x3=idx_3x3, pos='end')
    table3x3_T, table5x5_T = latexTableAnalysis(table3x3_T, table5x5_T, yTarget=yTarTime, OFxt=yOFTime, yPred=yPredTime, idx_3x3=idx_3x3, pos='end')
    
    fileOpenClose(f'{pathAmp}/latexTableBest_energy_train_{version}', 'close','w+', fileTableE, table5x5_E)
    fileOpenClose(f'{pathTime}/latexTableBest_time_train_{version}', 'close','w+', fileTableT, table5x5_T)
    
    #return table5x5_E

## ============================
def getMinEtaPhi(signalsList):
    """
        Some clusters has a non expected number of occurences in Eta
        and in Phi. Because of this it i necessary to find the correct
        number of eta and phi occurences to adjust the cluster size
    """
        ## The number of occurences in Eta coordinate indicates the number of Phi points
        ## The number of occurences in Phi coordinate indicates the number of Eta points
    nPhi = list(set( getOccurences(signalsList[:,1] )))
    nEta = list(set( getOccurences(signalsList[:,2] )))
    nE   = len( signalsList[:,0] )
    
    #prod = [x*y  for x in nEta for y in nPhi]
        ## return a list with: product, nEta, nPhi
    prod = list(chain.from_iterable((x*y, x, y) for x in nEta for y in nPhi ))
        ## find the product most close to the length(etruth) to reshape cluster into nPhi,nEta size    
    idxMin = prod.index( min(prod, key=lambda x:abs(x-nE) ))

    while prod[idxMin] < nE:
        prod[idxMin] = prod[idxMin]*2
        idxMin = prod.index( min(prod, key=lambda x:abs(x-nE) ))    
    
    #return nEta[idxMin], nPhi[idxMin]
    #return nEta[idxMin%len(nEta)], nPhi[idxMin%len(nPhi)]
    if len(prod) == 3:
        return prod[1], prod[2]
    else :
        return prod[idxMin+1], prod[idxMin+2]
    
## ============================
def getOccurences(signalList):
    """
        Pick up the number of occurences of each eta, phi point
        to adjust clusters where the len(E) is different of 
        nEta*nPhi from len(set(E[:,1])) for eta's or len(set(E[:,2]))
        for phi's
    """
    elemnts = set(signalList)
    itemOccur = []
    for i in elemnts:
        itemOccur.append(list(signalList).count(i))
        
    return (itemOccur)
    
## ============================
def getDictParam(myDict, pathDay, typeTrain, pathOFlztEbins, **kwargs):        
    
    model   = kwargs.get('model')
    signal  = kwargs.get('signal')
    save    = kwargs.get('save')
    show    = kwargs.get('show')
    energy  = kwargs.get('energy')
    idx_5x5 = kwargs.get('idx_5x5')
                 
    if save is None: save = False
    if show is None: show = False
    
    path     = '/eos/user/m/msandesd/Pesquisa/phd/crosstalk'
    source   = f'{path}/datasets/simulatedClusters'
    pathLZTamp = f'{source}/lzt/pkl/clus7x7_amplitudes'
    pathLZTsamp = f'{source}/lzt/pkl/clus7x7_samples'
    
    idx_7x7 = array(range(49))
    
    idx_3x3 = getIdxClus_mxn(idx_7x7.reshape(7,7), 3, 3)
    ij_cell = array(['-3,3' , '-2,3' , '-1,3' , '0,3' , '1,3' , '2,3' , '3,3' , 
                   '-3,2' , '-2,2' , '-1,2' , '0,2' , '1,2' , '2,2' , '3,2' , 
                   '-3,1' , '-2,1' , '-1,1' , '0,1' , '1,1' , '2,1' , '3,1' , 
                   '-3,0' , '-2,0' , '-1,0' , '0,0' , '1,0' , '2,0' , '3,0' , 
                   '-3,-1', '-2,-1', '-1,-1', '0,-1', '1,-1', '2,-1', '3,-1', 
                   '-3,-2', '-2,-2', '-1,-2', '0,-2', '1,-2', '2,-2', '3,-2', 
                   '-3,-3', '-2,-3', '-1,-3', '0,-3', '1,-3', '2,-3', '3,-3' ])
    
    if idx_5x5 is None: 
        idx_5x5 = getIdxClus_mxn(idx_7x7.reshape(7,7), 5, 5)
        
    ij_5x5  = ij_cell[idx_5x5]
    key = list(myDict)[0]
        
    InputDim  = myDict[key]['info']['InputDim']
    OutputDim = myDict[key]['info']['OutputDim']
    idxTest   = myDict[key]['DataIdx']['idxTest']
    batchSize = myDict[key]['info']['BatchSize']
    
    if  OutputDim  == 196 or OutputDim  == 49:
        clusCells = idx_7x7
        idxSampCells = array(range(49*4))
    elif  OutputDim  == 100 or OutputDim  == 25:
        clusCells = idx_5x5
        idxSampCells = getIdxSampClus( array(range(49*4)), 5 )
    else:        
        clusCells = idx_3x3
        idxSampCells = getIdxSampClus( array(range(49*4)), 3 )
    
    dictAmp  = loadSaveDict( glob.glob( f'{pathLZTamp}/{energy}/*Amplitudes*.pkl')[0], load=True)
    dictSamp = loadSaveDict( glob.glob( f'{pathLZTsamp}/{energy}/*Samples*.pkl')[0], load=True)
    xData    = np.add( np.add(np.add( dictSamp['E'], dictSamp['XT_C']), dictSamp['XT_L']), dictSamp['Noise'] )
    xTarg    = np.add( dictSamp['E'], dictSamp['Noise'] )
    scaler   = StandardScaler()
    xDataNorm = scaler.fit_transform(xData)
    xTargNorm = scaler.fit_transform(xTarg)
    ## =========================================
    ## --------------------------------
    ##         Target parameters
    ## --------------------------------
    TargOF   = OptFilt( xTarg )
    yTarAmp  = TargOF['Clusters']['Amplitude']
    yTarTime = TargOF['Clusters']['Time']
    nCell = yTarAmp.shape[1]
           
    pathTrue = createPath(f'{pathDay}/resume/plots')
       
    ## --------------------------------
    ##          OF parameters
    ## --------------------------------
    ## values with XT        
    OF = OptFilt( xData )
    
    yOFAmp  = OF['Clusters']['Amplitude']
    yOFTime = OF['Clusters']['Time']

    ## --------------------------------
    ##       Predict parameters
    ## --------------------------------
    ## Predicted values
    if typeTrain.lower() == 'samples':
        sampPred = model.predict(xDataNorm,#[np.ix_(idxTest, idxSampCells)],
                            batch_size = batchSize,
                            verbose    = 0,
                            workers    = 2,
                            use_multiprocessing=True)
        
        yPredAmp  = OptFilt( sampPred )['Clusters']['Amplitude']
        yPredTime = OptFilt( sampPred )['Clusters']['Time']

        nCells_targ = yTarAmp.shape[1]       
    else :
        if signal.lower() == 'energy':
            #signal = 'E'
            #yPredAmp = myDict[key]['Outputs']['yTarget']
            yPredTime = 0
            #nCells_targ = yTarAmp.shape[1]    
            
            yPredAmp = model.predict(xDataNorm,#[np.ix_(idxTest, idxSampCells)],
                                batch_size = batchSize,
                                verbose    = 0,
                                workers    = 2,
                                use_multiprocessing=True)
            
        else:
            #signal = r'$\tau$'
            order = 'ns'
            #yPredTime = myDict[key]['Outputs']['yTarget']
            yPredAmp = 0
            #nCells_targ = yTarTime.shape[1]    
            yPredTime = model.predict(xDataNorm,#[np.ix_(idxTest, idxSampCells)],
                                batch_size = batchSize,
                                verbose    = 0,
                                workers    = 2,
                                use_multiprocessing=True)
    ## =========================================

    yOFAmp   = yOFAmp[:, clusCells]
    yOFTime  = yOFTime[:, clusCells]
        
    sigPred = {'yPredAmp': yPredAmp, 'yPredTime':yPredTime } 
    
    if typeTrain.lower() == 'samples':
         
        sigEclusPred  = getSigmClus(yPredAmp, yTarAmp)    
        sigEcellsPred = getSigmClus(yPredAmp, yTarAmp)

        sigEclusOF    = getSigmClus(yOFAmp, yTarAmp)    
        sigEcellsOF   = getSigmClus(yOFAmp, yTarAmp)
        
        sigTclusPred  = getSigmCells(yPredTime, yTarTime)    
        sigTcellsPred = getSigmCells(yPredTime, yTarTime)
    
        sigTclusOF    = getSigmCells(yOFTime, yTarTime)    
        sigTcellsOF   = getSigmCells(yOFTime, yTarTime)
                
    elif typeTrain.lower() == 'energy':
        if np.max(np.mean(yTarAmp, axis=0)) > 9.99e3:
            order = 'GeV'
            yPredAmp = yPredAmp/1e3
            yOFAmp   = yOFAmp/1e3
            yTarAmp  = yTarAmp/1e3
        else :  order = 'MeV'
    
        sigEclusPred  = getSigmClus(yPredAmp, yTarAmp)    
        sigEcellsPred = getSigmClus(yPredAmp, yTarAmp)
    
        sigEclusOF    = getSigmClus(yOFAmp, yTarAmp)    
        sigEcellsOF   = getSigmClus(yOFAmp, yTarAmp)

    else :
        sigTclusPred  = getSigmCells(yPredTime, yTarTime)    
        sigTcellsPred = getSigmCells(yPredTime, yTarTime)

        sigTclusOF    = getSigmCells(yOFTime, yTarTime)    
        sigTcellsOF   = getSigmCells(yOFTime, yTarTime)
        
    sigTar = {'yTarAmp': yTarAmp, 'yTarTime':yTarTime } 
    sigOF  = {'yOFAmp': yOFAmp, 'yOFTime':yOFTime } 
        
    for idx, cell in enumerate( clusCells ):
        
        fileName  = f'True_{energy}_cell{ij_cell[cell]}_mean'
        titleName = r'$\tau$'+ f' - Target '+r'$\times$'+ r'$\tau_{{OF}}$ '+r'$\times$'+ r'$\tau_{{{struct}}}$ '+ f' | Cell {ij_cell[cell]}'      
        
        if typeTrain.lower() == 'samples':
            plotHisto( yOFAmp[:,idx], pathOut=pathTrue, label='Energy', legend=[f'Cell$_{{{ij_cell[cell]}}}$', r'Err$_{OF}$'], fileName=fileName,save=True, adjustXlim=False, show=show, MeanRms='mean', unit='GeV')
            plotHisto( yOFTime[:,idx], pathOut=pathTrue, label='Time', legend=[f'Cell$_{{{ij_cell[cell]}}}$', r'Err$_{OF}$'], fileName=fileName,save=True, adjustXlim=False, show=show, MeanRms='mean', unit='ns')
            order = ''
        else:
            
            if signal.lower() == 'energy': yOF = yOFAmp
            else: yOF = yOFTime

            plotHisto( yOF[:,idx], pathOut=pathTrue, label=signal, legend=[f'Cell$_{{{ij_cell[cell]}}}$', r'Err$_{OF}$'], fileName=fileName,save=save, show=show, adjustXlim=False, MeanRms='mean', unit=order)
        
    #return yTarAmp, yTarTime, yOFAmp, yOFTime, yPredAmp, yPredTime, order
    return {f'{energy}':{'yTarAmp':yTarAmp, 'yTarTime':yTarTime, 'yOFAmp':yOFAmp, 'yOFTime':yOFTime, 'yPredAmp':yPredAmp, 'yPredTime':yPredTime, 'order':order, 'nCell':nCell, 'Sigmas': {'sigPred': sigPred, 'sigTar':sigTar, 'sigOF': sigOF } }} #, 'rmse':{'Amp':mse(yTarAmp, yPredAmp, squared=False), 'Time':mse(yTarTime, yPredTime, squared=False)}}}

## ===========================
def topText(listCells, run, MeV_GeVns, rmse, eLabel, pred, of, targ, signal):
    listCells.append('.'.center(132, '.'))
    listCells.append(f'{eLabel} GeV - MLP net'.center(84, ' ')+ '|'+f'rmse'.center(22)+'|'+f'r2Score'.center(22))
    listCells.append('.'.center(132, '.'))
    space0, space1, space2, space3 = 7, 10, 26, 10
        
    if targ.shape[1]==25:
        idx3 = getIdxClus_mxn(array(range(25)).reshape(5,5), 3, 3)
        listCells.append(f'{run}'.center(84, ' ')+ '|'+f'{rmse:.4f}'.center(22)+ '|'+f'{r2_score(targ[:, idx3], pred[:, idx3]):.4f}'.center(22))
    else:
        listCells.append(f'{run}'.center(84, ' ')+ '|'+f'{rmse:.4f}'.center(22)+ '|'+f'{r2_score(targ[:, idx3x3], pred[:, idx3x3]):.4f}'.center(22))
    listCells.append('.'.center(132, '.'))

    listCells.append(f'Cell'.center(space0, ' ')+' & '+
                     f'Target [{MeV_GeVns}]'.center(space2, ' ')+ ' & ' +
                     f'OF [{MeV_GeVns}]'.center(space2, ' ')    + ' & ' + f'rmse'.center(space3, ' ') +' & '+
                     f'MLP [{MeV_GeVns}]'.center(space2, ' ')   + ' & ' + f'rmse'.center(space3, ' ') + ' \\\ ')

## ===========================
def middleText(listCells, pred, of, targ, idx, signal):
    space1, space2, space3 = 7, 9, 10
    #if signal.lower() == 'energy':
     #   pred, targ, of = pred/1000, targ/1000, of/1000
    
    listCells.append(f"{ ij_cell[idx]}".ljust(space1,' ') + " & " + f"{np.mean(targ[:, idx]):.2f}".rjust(space2,' ') + f" $\pm$ " + f"{np.std(targ[:, idx]):.2f}".rjust(space2,' ') + f" & " +
                     f"{np.mean(of[:, idx]):.2f}".rjust(space2, ' ')  + f" $\pm$ " + f"{np.std(of[:, idx]):.2f}".rjust(space2,' ')  + f" & " + f"{rms_e(of[:, idx], targ[:, idx]):.4f}".rjust(space3,' ') + f" & " +
                     f"{np.mean(pred[:, idx]):.2f}".rjust(space2,' ') + f" $\pm$ " + f"{np.std(pred[:, idx]):.2f}".rjust(space2,' ')+ f" & " + f"{rms_e(pred[:, idx], targ[:, idx]):.4f}".rjust(space3,' ') + '\\\ ')

    """
     '  Cell  &        Target [GeV]        &          OF [GeV]          &    rmse    &         MLP [GeV]          &    rmse    ',
     '-1,1    &       0.30$\\pm$      0.26 &       0.19 $\\pm$       0.27 &   0.0328 &       0.16 $\\pm$       0.26 &   0.3633',
     '0,1     &       0.87$\\pm$      0.63 &       0.64 $\\pm$       0.64 &   0.0453 &       0.59 $\\pm$       0.63 &   0.9009',
    """

## ===========================
def bottomText(listCells, pred, of, targ):
    if targ.shape[1]==25:
        idx3 = getIdxClus_mxn(array(range(25)).reshape(5,5), 3, 3)
        listCells.append(f'Cluster 3x3 '.center(65, ' ')+' & '+f"{rms_e(of[idx3], targ[idx3]):.4f}".center(12,' ')+' & ' +' '.center(26,' ')+' & '+f"{rms_e(pred[:, idx3], targ[:, idx3]):.4f}".rjust(10,' '))
    else:
        listCells.append(f'Cluster 3x3 '.center(65, ' ')+' & '+f"{rms_e(of[idx3x3], targ[idx3x3]):.4f}".center(10,' ')+' & ' +' '.center(26,' ')+' & '+f"{rms_e(pred[:, idx3x3], targ[:, idx3x3]):.4f}".rjust(10,' '))
    listCells.append(f'Cluster 5x5 '.center(65, ' ')+' & '+f"{rms_e(of, targ):.4f}".center(10,' ')+' & ' +' '.center(26,' ')+' & '+f"{rms_e(pred, targ):.4f}".rjust(10,' '))
    listCells.append('{:^100}'.format('*'*40))

## ===========================
def _getDictParam(myDict, pathDay, energy=None, save=False):
    source   = '/eos/user/m/msandesd/Pesquisa/phd/crosstalk/datasets/simulatedClusters'    
    lztData  = f'{source}/lzt/npy/clusters_e/energy_bins'

    ij_5x5 = ['-2,2',  '-1,2',  '0,2',  '1,2',  '2,2',  '-2,1',  '-1,1',  '0,1',  '1,1',  '2,1', '-2,0',  '-1,0',  '0,0',  '1,0',  '2,0', '-2,-1', '-1,-1', '0,-1', '1,-1', '2,-1', '-2,-2', '-1,-2', '0,-2', '1,-2', '2,-2']
    idx_3x3 = ['-1,1',  '0,1',  '1,1',  '-1,0',  '0,0',  '1,0',  '-1,-1', '0,-1', '1,-1', ]
    idx_3x3 = [6, 7, 8, 11, 12, 13, 16, 17, 18]
    
    key = list(myDict)[0]
        
    idxTest = myDict[key]['DataIdx']['idxTest']
    
    ampTime  = OptFilt(myDict[key]['Outputs']['yTarget'])
    yTarAmp  = ampTime['Clusters']['Amplitude']
    yTarTime = ampTime['Clusters']['Time']
    sigTar   = ampTime['Clusters']['Sigmas']
    
    pathTrue = createPath(f'{pathDay}/resume/plots')
       
    ## values with XT
    try:        
        yOFAmp  = myDict[key]['Outputs']['OF']['XT']['AmpClusters'][idxTest]
        yOFTime = myDict[key]['Outputs']['OF']['XT']['TimeClusters'][idxTest]            
    except:
        OF = loadSaveDict(glob.glob(f'{pathDay}/resume/OF*{energy}*.pkl')[0], load=True)
        yOFAmp  = OF['XT']['AmpClusters'][idxTest]
        yOFTime = OF['XT']['TimeClusters'][idxTest]

    sum_E_clus  = np.sum(yOFAmp, axis=1)
    std_E_clus  = np.std(sum_E_clus)    
    
    sigOF = {'Sigma_Over_E': {'Cells': np.std( yOFAmp, axis=0 )/np.mean(yOFAmp, axis=0), 
                              'Clus': std_E_clus/np.mean(sum_E_clus) } , 
             'Sigma_Tau': {'Cells': np.std( yOFTime, axis=0 ),
                           'Clus': np.std(np.mean( yOFTime, axis=0 )) } } 
        
    ## Predicted values
    ampTime   = OptFilt(myDict[key]['Outputs']['yPred'])
    yPredAmp  = ampTime['Clusters']['Amplitude']
    yPredTime = ampTime['Clusters']['Time']
    sigPred   = ampTime['Clusters']['Sigmas']

    if np.max(np.mean(yTarAmp, axis=0)) > 9.99e3:
        order = 'GeV'
        yPredAmp = yPredAmp/1e3
        yOFAmp   = yOFAmp/1e3
        yTarAmp  = yTarAmp/1e3
    else :  order = 'MeV'
        
    ij = yOFAmp.shape[1]
    if ij == 9: clus_Size = idx_3x3
    else : clus_Size = np.linspace(0, 24, 25, dtype=int)
        
    for cell in clus_Size:
        fileName  = f'True_{energy}_cell{ij_5x5[cell]}_mean'
        titleName = r'$\tau$'+ f' - Target '+r'$\times$'+ r'$\tau_{{OF}}$ '+r'$\times$'+ r'$\tau_{{{struct}}}$ '+ f' | Cell {ij_5x5[cell]}'

        plotHisto( yOFAmp[:,cell], pathOut=pathTrue, label='Energy', legend=[f'Cell$_{{{ij_5x5[cell]}}}$', r'Err$_{OF}$'], fileName=fileName,save=True, adjustXlim=False, MeanRms='mean', unit=order)
        
    #return yTarAmp, yTarTime, yOFAmp, yOFTime, yPredAmp, yPredTime, order
    return {f'{energy}':{'yTarAmp':yTarAmp, 'yTarTime':yTarTime, 'yOFAmp':yOFAmp, 'yOFTime':yOFTime, 'yPredAmp':yPredAmp, 'yPredTime':yPredTime, 'order':order, 'Sigmas': {'sigPred': sigPred, 'sigTar':sigTar, 'sigOF': sigOF } }} #, 'rmse':{'Amp':mse(yTarAmp, yPredAmp, squared=False), 'Time':mse(yTarTime, yPredTime, squared=False)}}}


## =========================================    
def _getDictParam(myDict, pathDay, energy=None, save=False):
    ij_5x5 = ['-2,2',  '-1,2',  '0,2',  '1,2',  '2,2',  '-2,1',  '-1,1',  '0,1',  '1,1',  '2,1', '-2,0',  '-1,0',  '0,0',  '1,0',  '2,0', '-2,-1', '-1,-1', '0,-1', '1,-1', '2,-1', '-2,-2', '-1,-2', '0,-2', '1,-2', '2,-2']
    idx_3x3 = ['-1,1',  '0,1',  '1,1',  '-1,0',  '0,0',  '1,0',  '-1,-1', '0,-1', '1,-1', ]
    idx_3x3 = [6, 7, 8, 11, 12, 13, 16, 17, 18]
    
    key = list(myDict)[0]
        
    idxTest = myDict[key]['DataIdx']['idxTest']
    
    ## Target values
    ampTime  = OptFilt(myDict[key]['Outputs']['yTarget'])
    yTarAmp  = ampTime['Clusters']['Amplitude']
    yTarTime = ampTime['Clusters']['Time']
    
    pathTrue = createPath(f'{pathDay}/resume/plots')
       
    ## values with XT
    try:        
        yOFAmp  = myDict[key]['Outputs']['OF']['XT']['AmpClusters'][idxTest]
        yOFTime = myDict[key]['Outputs']['OF']['XT']['TimeClusters'][idxTest]    
    except:
        OF = loadSaveDict(glob.glob(f'{pathDay}/resume/OF*{energy}*.pkl')[0], load=True)
        yOFAmp  = OF['XT']['AmpClusters'][idxTest]
        yOFTime = OF['XT']['TimeClusters'][idxTest]
        
    ij = yOFAmp.shape[1]
    if ij == 9: clus_Size = idx_3x3
    else : clus_Size = np.linspace(0, 24, 25, dtype=int)
        
    for cell in clus_Size:
        fileName  = f'True_{energy}_cell{ij_5x5[cell]}_mean'
        titleName = r'$\tau$'+ f' - Target '+r'$\times$'+ r'$\tau_{{OF}}$ '+r'$\times$'+ r'$\tau_{{{struct}}}$ '+ f' | Cell {ij_5x5[cell]}'

        plotHisto( yOFAmp[:,cell], pathOut=pathTrue, label='Energy', legend=[f'Cell$_{{{ij_5x5[cell]}}}$', r'Err$_{OF}$'], fileName=fileName,save=save, adjustXlim=False, MeanRms='mean', unit='MeV')
        
    ## Predicted values
    ampTime   = OptFilt(myDict[key]['Outputs']['yPred'])
    yPredAmp  = ampTime['Clusters']['Amplitude']
    yPredTime = ampTime['Clusters']['Time']

    if np.max(np.mean(yTarAmp, axis=0)) > 9.99e3:
        order = 'GeV'
        yPredAmp = yPredAmp/1e3
        yOFAmp   = yOFAmp/1e3
        yTarAmp  = yTarAmp/1e3
        
    else :  order = 'MeV'
        
    return yTarAmp, yTarTime, yOFAmp, yOFTime, yPredAmp, yPredTime, order

## =========================================================
def printInfo( **kwargs ):
    signal = kwargs.get('signal')
    lay    = kwargs.get('lay')
    energy = kwargs.get('energy')    
    init   = kwargs.get('init')
    nInit  = kwargs.get('nInit')
    sufix  = kwargs.get('sufix')
    nFolds = kwargs.get('nFolds')
    fold   = kwargs.get('fold')
    regul  = kwargs.get('regul')
    nNeu   = kwargs.get('nNeu')
    
    print('{:^60}'.format("*"*30))
    print('{:*^60}'.format(' '+str(signal)+' - regression task. '))
    if energy is not None:
        print('{:.^60}'.format(f' {energy} '))
    #print('{:^60}'.format(' Neurons: '+str(lay)+ ' | Layers: '+ len(lay)+' '))
    print('{:^60}'.format(f' Layers: {len(list(lay))} | Neurons: {str(lay).replace("["," ").replace("]"," ")} '))
    print('{:^60}'.format('Fold '+str(fold)+f'/{nFolds}'))
    print('{:^60}'.format( "_"*34))
    print('{:^60}'.format(' Initialization '+str(init+1)+'/'+str(nInit)+' '))
    print('{:^60}'.format( "_"*34))
    
    if regul is None: 
        print('{:-^60}'.format(' Training model ... ')) 
    else : 
        print('{:-^60}'.format(' Training model with Regularizer... ')) 
        
    return

##==================================
def latexTableAnalysis(table3x3, table5x5, **kwargs):
    run     = kwargs.get('run')
    rmse    = kwargs.get('rmse')
    yTarget = kwargs.get('yTarget')
    yPred   = kwargs.get('yPred')
    OFxt    = kwargs.get('OFxt')
    pos     = kwargs.get('pos')
    cell    = kwargs.get('cell')
    MeV_GeVns = kwargs.get('MeV_GeVns')
    idx_5x5 = kwargs.get('idx_5x5')
    idx     = kwargs.get('idx')            

    idx_7x7 = array(range(49))
    idx_5x5 = getIdxClus_mxn(idx_7x7.reshape(7,7), 5, 5)
    idx_3x3 = getIdxClus_mxn(idx_7x7.reshape(7,7), 3, 3)
    ij_cell = ['-3,3' , '-2,3' , '-1,3' , '0,3' , '1,3' , '2,3' , '3,3' , 
               '-3,2' , '-2,2' , '-1,2' , '0,2' , '1,2' , '2,2' , '3,2' , 
               '-3,1' , '-2,1' , '-1,1' , '0,1' , '1,1' , '2,1' , '3,1' , 
               '-3,0' , '-2,0' , '-1,0' , '0,0' , '1,0' , '2,0' , '3,0' , 
               '-3,-1', '-2,-1', '-1,-1', '0,-1', '1,-1', '2,-1', '3,-1', 
               '-3,-2', '-2,-2', '-1,-2', '0,-2', '1,-2', '2,-2', '3,-2', 
               '-3,-3', '-2,-3', '-1,-3', '0,-3', '1,-3', '2,-3', '3,-3' ]

    #print(f'yTarget: {yTarget.shape} | yPred: {yPred.shape} | rsm: {mse(yTarget[:, idx_5x5, yPred)}' )
    if OFxt is None: OFxt = 0    
    ## ======================
    ## -------- BEGIN -------
    if pos == 'begin':        
        if list(idx_3x3).count(idx) > 0:
            table3x3.append('.'.center(120, '.')+'\n')
            table3x3.append(f'MLP net'.center(74, ' ')+ '|'+f'rmse'.center(18)+'|'+f'r2Score'.center(18)+'\n')
            table3x3.append(f'{run}'.center(74, ' ')+ '|'+f'{rmse:.4e}'.center(18)+ '|'+f'{r2_score(yTarget[:, idx_3x3], yPred[:, idx_3x3]):.4e}'.center(18)+'\n')
            table3x3.append('.'.center(120, '.')+'\n')

            table3x3.append(f' '.center(12, ' ')+'&'+f' Target [{MeV_GeVns}]'.center(25, ' ')+'&'+f' OF [{MeV_GeVns}] '.center(25, ' ')+ '&'+ f' rmse '.center(10, ' ') +'&'+f' MLP [{MeV_GeVns}] '.center(25, ' ')+'&'+f' rmse '.center(10, ' ') +'\n')

        table5x5.append('.'.center(120, '.')+'\n')
        table5x5.append(f'MLP net'.center(74, ' ')+ '|'+f'rmse'.center(18)+'|'+f'r2Score'.center(18)+'\n')
        table5x5.append(f'{run}'.center(74, ' ')+ '|'+f'{rmse:.4e}'.center(18)+ '|'+f'{r2_score(yTarget[:, idx_5x5], yPred[:, idx_5x5]):.4e}'.center(18)+'\n')
        table5x5.append('.'.center(120, '.')+'\n')

        table5x5.append(f' '.center(12, ' ')+'&'+f' Target [{MeV_GeVns}]'.center(25, ' ')+'&'+f' OF [{MeV_GeVns}] '.center(25, ' ')+ '&'+ f' rmse '.center(10, ' ') +'&'+f' MLP [{MeV_GeVns}] '.center(25, ' ')+'&'+f' rmse '.center(10, ' ') +'\n')
    ## ======================
    ## -------- MIDDLE ------
    elif pos == 'middle':
        if list(idx_3x3).count(idx) > 0:
            table3x3.append(f"Cell { cell } ".ljust(12,' ')+"& "+f"{rms(yTarget[:, idx]):.2f}".rjust(8,' ')+f" $\pm$ "+f"{np.std(yTarget[:, idx]):.2f}".rjust(8,' ')+f" & "+f"{np.mean(OFxt[:, idx]):.2f}".rjust(8, ' ')+f" $\pm$ "+f"{np.std(OFxt[:, idx]):.2e}".rjust(8,' ')+f" & "+f"{rms_e(OFxt[:, idx], yTarget[:, idx]):.2f}".rjust(8,' ')+f" & "+f"{np.mean(yPred[:, idx]):.2f}".rjust(8,' ')+f" $\pm$ "+f"{np.std(yPred[:, idx]):.2f}".rjust(8,' ')+f" & "+f"{rms_e(yPred[:, idx], yTarget[:, idx]):.2e}".rjust(8,' ')+' \\'+'\\'+'\n')
            
        table5x5.append(f"Cell { cell } ".ljust(12,' ')+"& "+f"{np.mean(yTarget[:, idx]):.2f}".rjust(8,' ')+f" $\pm$ "+f"{np.std(yTarget[:, idx]):.2f}".rjust(8,' ')+f" & "+f"{np.mean(OFxt[:, idx]):.2f}".rjust(8, ' ')+f" $\pm$ "+f"{np.std(OFxt[:, idx]):.2e}".rjust(8,' ')+f" & "+f"{rms_e(OFxt[:, idx], yTarget[:, idx]):.2f}".rjust(8,' ')+f" & "+f"{np.mean(yPred[:, idx]):.2f}".rjust(8,' ')+f" $\pm$ "+f"{np.std(yPred[:, idx]):.2f}".rjust(8,' ')+f" & "+f"{rms_e(yPred[:, idx], yTarget[:, idx]):.2e}".rjust(8,' ')+' \\'+'\\'+'\n')
    ## ======================
    ## -------- END ---------
    else :
        if list(idx_3x3).count(idx) > 0:
            table3x3.append(f'Cluster 3x3 '.center(60, ' ')+'&'+f"{rms_e(OFxt[idx_3x3], yTarget[idx_3x3]):.2e}".center(10,' ')+'&' +' '.center(25,' ')+'&'+f"{rms_e(yPred[:, idx_3x3], yTarget[:, idx_3x3]):.2e}".center(10,' ')+'\n')
            table3x3.append(f'Cluster 5x5 '.center(60, ' ')+'&'+f"{rms_e(OFxt, yTarget):.2e}".center(10,' ')+'&' +' '.center(25,' ')+'&'+f"{rms_e(yPred, yTarget):.2e}".center(10,' ')+'\n')
            table3x3.append('{:^100}'.format('*'*40)+'\n\n')

        table5x5.append(f'Cluster 3x3 '.center(60, ' ')+'&'+f"{rms_e(OFxt[:, idx_3x3], yTarget[:, idx_3x3]):.2e}".center(10,' ')+'&' +' '.center(25,' ')+'&'+f"{rms_e(yPred[:,idx_3x3], yTarget[:, idx_3x3]):.2e}".center(10,' ')+'\n')
        table5x5.append(f'Cluster 5x5 '.center(60, ' ')+'&'+f"{rms_e(OFxt[:, idx_5x5], yTarget[:, idx_5x5]):.2e}".center(10,' ')+'&' +' '.center(25,' ')+'&'+f"{rms_e(yPred[:, idx_5x5], yTarget[:, idx_5x5]):.2e}".center(10,' ')+'\n')
        table5x5.append('{:^100}'.format('*'*40)+'\n\n\n\n')
            
        table5x5 = table5x5 + table3x3
            
    return table3x3, table5x5

## =====================================================
### Export Table from Training

def _latexTableAnalysis(table3x3, table5x5, **kwargs):
    run     = kwargs.get('run')
    rmse    = kwargs.get('rmse')
    yTarget = kwargs.get('yTarget')
    yPred   = kwargs.get('yPred')
    OFxt    = kwargs.get('OFxt')
    pos     = kwargs.get('pos')
    cell    = kwargs.get('cell')
    MeV_GeVns = kwargs.get('MeV_GeVns')
    idx_3x3 = kwargs.get('idx_3x3')
    idx     = kwargs.get('idx')        
    
    idx_3x3 = [6, 7, 8, 11, 12, 13, 16, 17, 18]
    
    if OFxt is None: OFxt = 0
    
    #print(f'inside latexTableAnalysis: {idx}')
    
    #if idx_3x3.count(idx) > 0:
        #print(f'inside idx_3x3: {idx}')
    #if idx_3x3.count(i) > 0:
    if pos == 'begin':
        if idx_3x3.count(idx) > 0:
            table3x3.append('.'.center(120, '.')+'\n')
            table3x3.append(f'MLP net'.center(74, ' ')+ '|'+f'rmse'.center(18)+'|'+f'r2Score'.center(18)+'\n')
            table3x3.append(f'{run}'.center(74, ' ')+ '|'+f'{rmse:.4e}'.center(18)+ '|'+f'{r2_score(yTarget, yPred):.4e}'.center(18)+'\n')
            table3x3.append('.'.center(120, '.')+'\n')

            table3x3.append(f' '.center(18, ' ')+'&'+f' Target [{MeV_GeVns}]'.center(25, ' ')+'&'+f' OF [{MeV_GeVns}] '.center(25, ' ')+ '&'+ f' rmse '.center(10, ' ') +'&'+f' MLP [{MeV_GeVns}] '.center(25, ' ')+'&'+f' rmse '.center(10, ' ') +'\n')

        table5x5.append('.'.center(120, '.')+'\n')
        table5x5.append(f'MLP net'.center(74, ' ')+ '|'+f'rmse'.center(18)+'|'+f'r2Score'.center(18)+'\n')
        table5x5.append(f'{run}'.center(74, ' ')+ '|'+f'{rmse:.4e}'.center(18)+ '|'+f'{r2_score(yTarget, yPred):.4e}'.center(18)+'\n')
        table5x5.append('.'.center(120, '.')+'\n')

        table5x5.append(f' '.center(18, ' ')+'&'+f' Target [{MeV_GeVns}]'.center(25, ' ')+'&'+f' OF [{MeV_GeVns}] '.center(25, ' ')+ '&'+ f' rmse '.center(10, ' ') +'&'+f' MLP [{MeV_GeVns}] '.center(25, ' ')+'&'+f' rmse '.center(10, ' ') +'\n')
    elif pos == 'middle':
        if idx_3x3.count(idx) > 0:
            table3x3.append(f"Cell$_{{ { cell }}}$ ".ljust(18,' ')+"& "+f"{rms(yTarget):.2f}".rjust(8,' ')+f" $\pm$ "+f"{np.std(yTarget):.2f}".rjust(8,' ')+f" & "+f"{rms(OFxt):.2f}".rjust(8, ' ')+f" $\pm$ "+f"{np.std(OFxt):.2e}".rjust(8,' ')+f" & "+f"{rms_e(OFxt, yTarget):.2e}".rjust(8,' ')+f" & "+f"{rms(yPred):.2e}".rjust(8,' ')+f" $\pm$ "+f"{np.std(yPred):.2f}".rjust(8,' ')+f" & "+f"{rms_e(yPred, yTarget):.2e}".rjust(8,' ')+' \\'+'\\'+'\n')
            
        table5x5.append(f"Cell$_{{ { cell }}}$ ".ljust(18,' ')+"& "+f"{rms(yTarget):.2e}".rjust(8,' ')+f" $\pm$ "+f"{np.std(yTarget):.2f}".rjust(8,' ')+f" & "+f"{rms(OFxt):.2f}".rjust(8, ' ')+f" $\pm$ "+f"{np.std(OFxt):.2e}".rjust(8,' ')+f" & "+f"{rms_e(OFxt, yTarget):.2e}".rjust(8,' ')+f" & "+f"{rms(yPred):.2e}".rjust(8,' ')+f" $\pm$ "+f"{np.std(yPred):.2f}".rjust(8,' ')+f" & "+f"{rms_e(yPred, yTarget):.2e}".rjust(8,' ')+' \\'+'\\'+'\n')
        
    else :
        if idx_3x3.count(idx) > 0:
            table3x3.append(f'Cluster 3x3 '.center(70, ' ')+'&'+f"{rms_e(OFxt[idx_3x3], yTarget[idx_3x3]):.2e}".center(10,' ')+'&' +' '.center(25,' ')+'&'+f"{rms_e(yPred[idx_3x3], yTarget[idx_3x3]):.2e}".center(10,' ')+'\n')
            table3x3.append(f'Cluster 5x5 '.center(70, ' ')+'&'+f"{rms_e(OFxt, yTarget):.2e}".center(10,' ')+'&' +' '.center(25,' ')+'&'+f"{rms_e(yPred, yTarget):.2e}".center(10,' ')+'\n')
            table3x3.append('{:^100}'.format('*'*40)+'\n\n')

        table3x3.append(f'Cluster 3x3 '.center(70, ' ')+'&'+f"{rms_e(OFxt[idx_3x3], yTarget[idx_3x3]):.2e}".center(10,' ')+'&' +' '.center(25,' ')+'&'+f"{rms_e(yPred[idx_3x3], yTarget[idx_3x3]):.2e}".center(10,' ')+'\n')
        table5x5.append(f'Cluster 5x5 '.center(70, ' ')+'&'+f"{rms_e(OFxt, yTarget):.2e}".center(10,' ')+'&' +' '.center(25,' ')+'&'+f"{rms_e(yPred, yTarget):.2e}".center(10,' ')+'\n')
        table5x5.append('{:^100}'.format('*'*40)+'\n\n\n')
            
        table5x5 = table5x5 + table3x3
            
    return table3x3, table5x5

## =========================================================
def getMeanRms(x1, MeanRms='mean'):    

    # Determine whether to calculate 'mean' or 'rms'
    if MeanRms.lower() == 'mean':
        means = np.mean(x1, axis=1)  # Calculate mean for each column
        #stds = np.std(x1, axis=1)    # Calculate std for each column
        stds = np.maximum(np.round(np.std(x1, axis=1), 2), 0.01) 
        
    elif MeanRms.lower() == 'rms':
        means = rms(x1, axis=1)      # Calculate rms for each column
        #stds = np.std(x1, axis=1)    # Calculate std for each column
        stds = np.maximum(np.round(np.std(x1, axis=1), 2), 0.01)    # Calculate std for each column
        
        
    else:
        raise ValueError("MeanRms should be either 'mean' or 'rms'.")

    # Intercalate means and stds dynamically
    results = []
    for mean, std in zip(means, stds):
        results.append([mean, std])
        #results.append(std)

    return results


## =========================================================
def min_max(x1):
    """
        x1 is list of arrays
    """
    return np.min(x1), np.max(x1)

##==========================================================
def modelConfigs(model, earlyStop, pathFold, xTrain, yTrain, Nepochs, SizeBatchFit, xVal, yVal):
    es = EarlyStopping(monitor    = 'val_loss',
                        mode      = 'min',
                        min_delta = 0.001,
                        patience  = earlyStop)
    mc = ModelCheckpoint(f'{pathFold}/best_model', 
                            monitor = 'val_loss',
                            mode    = 'min',
                            verbose = 0,
                            save_best_only=True)
    history = model.fit(xTrain, yTrain,
                        epochs     = Nepochs,
                        batch_size = SizeBatchFit,
                        validation_data = (xVal, yVal),
                        callbacks  = [es, mc],
                        verbose    = 2)#,
                        #workers    = 2,
                        #use_multiprocessing=True)
    return history

##==========================================================
def modelPredict(model, xTrain, xVal, xTest, xData, SizeBatch):
    yTrainPredic = model.predict(xTrain, batch_size=SizeBatch)
    yValPredic = model.predict(xVal, batch_size=SizeBatch)
    yTestPredic = model.predict(xTest, batch_size=SizeBatch)
    yDataPredic = model.predict(xData, batch_size=SizeBatch)

    return yTrainPredic, yValPredic, yTestPredic, yDataPredic

##==========================================================
def modelEvaluate(model, xTrain, yTrain, xVal, yVal, xData, yData, xTest, yTest, SizeBatch):
    EvalTrain = model.evaluate(xTrain,
                                yTrain, 
                                batch_size = SizeBatch,
                                verbose = 1)#,
                                #workers = cores,
                                #use_multiprocessing=True)
                                #return_dict=True)
    EvalVal   = model.evaluate(xVal,
                                yVal, 
                                batch_size = SizeBatch,
                                verbose = 1)#,
                                #workers = cores)#,
                                #use_multiprocessing=True)
                                # return_dict=True)
    EvalxData  = model.evaluate(xData,
                                yData, 
                                batch_size = SizeBatch,
                                verbose = 0)#,
                                #workers = cores,
                                #use_multiprocessing=True)
    EvalTest   = model.evaluate(xTest,
                                yTest, 
                                batch_size = SizeBatch,
                                verbose = 1)#,
                                #workers = cores,
                                #use_multiprocessing=True)
                                #return_dict=True)  
    return EvalTrain, EvalVal, EvalxData, EvalTest
##===========================================================
def recEnergyTimeNoNoise(Esamples, gSignal, DgSignal):
    EnergyNoNoise = np.zeros([Esamples.shape[0],9])
    TimeNoNoise   = np.zeros([Esamples.shape[0],9])
    
    for samp in range(Esamples.shape[0]):
        for cell in range(9):
            EnergyNoNoise[samp, cell] = np.mean(Esamples[samp, 0+4*cell:4+4*cell]/gSignal[samp, 0+4*cell:4+4*cell])
            ## This expression has more three options, because this is the solution for a linear system
            ## S(t)  = E.g(t - tau) = E.g(t) - E.g'(t).tau => From Taylos 
            ## S(t1) = E.g(t1) - E.g'(t1).tau
            ## S(t2) = E.g(t2) - E.g'(t2).tau
            ## S(t3) = E.g(t3) - E.g'(t3).tau
            ## S(t4) = E.g(t4) - E.g'(t4).tau

            TimeNoNoise[samp, cell]   = (gSignal[samp, 4*cell + 3]*gSignal[samp,  4*cell + 2]*
                                        gSignal[samp,  4*cell + 1]*Esamples[samp, 4*cell + 0] - 
                                        gSignal[samp,  4*cell + 3]*gSignal[samp,  4*cell + 2]*
                                        gSignal[samp,  4*cell + 0]*Esamples[samp, 4*cell + 1])/((
                                        gSignal[samp,  4*cell + 3]*gSignal[samp,  4*cell + 2]*
                                        gSignal[samp,  4*cell + 1]*DgSignal[samp, 4*cell + 0] - 
                                        gSignal[samp,  4*cell + 3]*gSignal[samp,  4*cell + 2]*
                                        gSignal[samp,  4*cell + 0]*DgSignal[samp, 4*cell + 1])*
                                        EnergyNoNoise[samp, cell])
    return EnergyNoNoise, TimeNoNoise

##===========================================================
### Sort array in ascending/descending order
def sortArray(MinMax, nArray, n=None):
    if n is None:
        n = len(nArray)
    nArray = deepcopy(nArray)
    valList, idxList = np.zeros([n]), np.zeros([n],dtype=int)
    for i in range(n):
        if MinMax == 'min':
            valList[i], idxList[i] = min((val, idx) for idx, val in enumerate(nArray))
            nArray[idxList[i]]     = np.max(nArray)*1e2
        elif MinMax == 'max':
            valList[i], idxList[i] = max((val, idx) for idx, val in enumerate(nArray))
            nArray[idxList[i]]     = 0
    return valList, idxList   

##===========================================================
### Export Table from Training

def tableTrain0(y, yRec, OF, idx, signal, struct, clus3x3=False):
    if clus3x3 is False:
        ij_cell = ['00', '01', '02', '03', '04', '10', '11', '12', '13', '14', '20', '21', '22', '23', '24', '30', '31', '32', '33', '34', '40', '41', '42', '43', '44']
    else :
        ij_cell = ['11', '12', '13', '21', '22', '23', '31', '32', '33']
    if signal == 'Time':
        EnergyTime = 'Time'
    else : 
        EnergyTime = 'Amplitude'
    content = []
    content.append(f'{" "*8}&{" "*11}yRef{" "*8} & {" "*11}OF{" "*10} &  RMSEof  & {" "*9}{struct}{" "*12} & RMSEnn \n')
    for cell in range(yRec.shape[1]):
        content.append(f'Cell {cell+1} & {rms(y[:, cell]):.3e} $\pm$ {rmsErr(y[:, cell]):.3e} & {rms(OF[f"Cell {cell+1}"][f"{EnergyTime}"][idx]):.3e} $\pm$ {rmsErr(OF[f"Cell {cell+1}"][f"{EnergyTime}"][idx]):.3e} & {mse(y[:, cell], OF[f"Cell {cell+1}"][f"{EnergyTime}"][idx],squared=False):.3e} & {rms(yRec[:, cell]):.3e} $\pm$ {rmsErr(yRec[:, cell]):.3e} & {mse(y[:, cell], yRec[:, cell],squared=False):.3e} \n')
    return content

##===========================================================
def timeInfo(time):
    hours   = int(time/3600)
    minutes =  int((time%3600)/60)
    seconds  = int((time%3600)%60)
    ms      = int(str(f'{time:.3f}')[-3:])
    
    if hours == 0: 
        if minutes == 0 :
            if seconds == 0:
                output = f'{ms} ms'
            else :
                output = f'{seconds} sec {ms} ms'
        else: 
            output = f'{minutes} min {seconds} sec {ms} ms'
    else : 
        output =f'{hours} h {minutes} min {seconds} sec {ms} ms'
        
    return output 

##===========================================================
### Export Table from Training

def tableTrain(y, yRec, OF, idx, signal, struct, xTest=False, Target=False, ai=False, bi=False):
    ij_cell = ['00', '01', '02', '03', '04', '10', '11', '12', '13', '14', '20', '21', '22', '23', '24', '30', '31', '32', '33', '34', '40', '41', '42', '43', '44']  
    content = []
    if y.shape[1] == 100:
        if signal == 'Time':
            EnergyTime = 'Time'
        else : 
            EnergyTime = 'Amplitude'    
        AmpTimeTest  = OptFilt( xTest, ai, bi )
        AmpTimePred  = OptFilt( yRec, ai, bi )
        #for cell in range(len(AmpTimeTest)):
        for cell in range(25):
            content.append(f'Cell {ij_cell[cell]} & {rms(Target[:, cell]):.3e} $\pm$ {rmsErr(Target[:, cell]):.3e} & {rms(AmpTimeTest[f"Cell { ij_cell[cell] }"][f"{EnergyTime}"]):.3e} $\pm$ {rmsErr(AmpTimeTest[f"Cell {ij_cell[cell]}"][f"{EnergyTime}"]):.3e} & {mse(Target[:, cell], AmpTimeTest[f"Cell {ij_cell[cell]}"][f"{EnergyTime}"],squared=False):.3e} & {rms(AmpTimePred[f"Cell {ij_cell[cell]}"][f"{EnergyTime}"]):.3e} $\pm$ {rmsErr(AmpTimePred[f"Cell { ij_cell[cell] }"][f"{EnergyTime}"]):.3e} & {mse(Target[:, cell], AmpTimePred[f"Cell { ij_cell[cell] }"][f"{EnergyTime}"],squared=False):.3e} \n')

        return content

    if y.shape[1] == 25 or y.shape[1] == 36:
        if signal == 'Time':
            EnergyTime = 'Time'
        else : 
            EnergyTime = 'Amplitude'        
        content.append(f'{" "*8}&{" "*11}yRef{" "*8} & {" "*11}OF{" "*10} &  RMSEof  & {" "*9}{struct}{" "*12} & RMSEnn \n')
        for cell in range(yRec.shape[1]):
            #content.append(f'Cell { ij_cell[cell] } & {rms(y[:, cell]):.3e} $\pm$ {rmsErr(y[:, cell]):.3e} & {rms(OF[f"Cell { ij_cell[cell] }"][f"{EnergyTime}"][idx]):.3e} $\pm$ {rmsErr(OF[f"Cell { ij_cell[cell] }"][f"{EnergyTime}"][idx]):.3e} & {mse(y[:, cell], OF[f"Cell { ij_cell[cell] }"][f"{EnergyTime}"][idx],squared=False):.3e} & {rms(yRec[:, cell]):.3e} $\pm$ {rmsErr(yRec[:, cell]):.3e} & {mse(y[:, cell], yRec[:, cell],squared=False):.3e} \n')
            content.append(f'Cell { cell } & {rms(y[:, idx]):.3e} $\pm$ {np.std(y[:, idx]):.3e} & {rms(OF[f"Cell { cell }"][EnergyTime]/1e3):.3e} $\pm$ {rms_e(OF[f"Cell { cell }"][EnergyTime]/1e3, y[:,idx]/1e3):.3e} & {mse(y[:, idx], OF[f"Cell { cell }"][EnergyTime]/1e3, squared=False):.3e} & {rms(yRec[:, idx]):.3e} $\pm$ {rms_e(yRec[:, idx], y[:,idx]):.3e} \n')
        return content


##===========================================================
### Convolutional Autoencoder
def plot_training_error(hList=None,model_type=None,ae=None,version=None,log=False,save_path=False,path=None, ext=None):
    #fig, axs = plt.subplots(nrows=1, ncols=3,figsize=(20,6))
    #fig, axs = plt.subplots(nrows=2, ncols=5,figsize=(20,6))
    fig, axs = plt.subplots(nrows=5, ncols=2,figsize=(15,20))
    
    #nrows=1, ncols=3: parametros para definir número de graficos relativos aos folds

    ax = axs.ravel()
    for idx,hist in enumerate(hList):
        fold=idx+1
        if log:
            ax[idx].set_yscale('log')
        try:
            ax[idx].plot(np.sqrt(hist.history['loss']))
            ax[idx].plot(np.sqrt(hist.history['val_loss']))
        except :
            ax[idx].plot(np.sqrt(hist['loss']))
            ax[idx].plot(np.sqrt(hist['val_loss']))

        
        ax[idx].set_title(f'Fold{fold}')
        ax[idx].set_ylabel('rmse')
        ax[idx].set_xlabel('epoch')
        ax[idx].legend(['train', 'val'], loc='upper right')
    #plt.suptitle(model_type+'-autoencoder-'+ae+'-V'+version,y=1.05)
    #plt.suptitle(f'{model_type}-autoencoder-{ae}-V{version}',y=1.05)
    plt.tight_layout()    
    if save_path:
        plt.savefig(path,format=ext, dpi=800)
        
##===========================================================
def rms(x, axis=None):
    return np.sqrt(np.mean(x**2, axis=axis))  

##===========================================================
def rms_e(y_pred, y_true, **kwargs):
    squared     = kwargs.get('squared')
    multioutput = kwargs.get('multioutput')
            
    #if multioutput is not None: multioutput='raw_values'
        
    return mse(y_true, y_pred, multioutput=multioutput, squared=False)

#def rmse(y_pred, y_true, **kwargs):
#    squared     = kwargs.get('squared')
#    multioutput = kwargs.get('squared')
            
#    if multioutput is None: multioutput='raw_values'
        
#    return mse(y_true, y_pred, multioutput=multioutput, squared=False)

#def rms(x):
#    if (x.ndim > 1):
#        return np.sqrt(np.mean(x**2, axis=0))
#    else :
#        return np.sqrt(np.mean(x**2))
##===========================================================
def rmsErr(x):
    if (x.ndim > 1):
        #return np.sqrt(np.mean((x-np.mean(x, axis=0))**2, axis=0))
        return np.sqrt(np.mean((x-np.mean(x, axis=1))**2, axis=1))
    else :
        return np.sqrt(np.mean((x-np.mean(x))**2))

##===========================================================
def rmseLoss(y_true, y_pred, multioutput=False):
    
    loss =  K.sqrt(K.mean(K.square( y_true - y_pred )))

    return loss


##===========================================================
"""def tStudentTest(Pa, Pb, k):
    ## From Kuncheva -  Combining Pattern Calssifiers
    Pa_mean = np.mean(Pa) 
    Pb_mean = np.mean(Pb) 
    P = Pa - Pb
    Pa_x_Pb = Pmlp - Pelm
    PmELMxMLP = mean(Pmlp - Pelm)
    tELMxMLP(4*(et-1)+eta) = PmELMxMLP*sqrt(k)/sqrt(sum((ELMxMLP-repmat(PmELMxMLP,50,1)).^2)/(k-1));
    return"""
#
##===========================================================
def wInitContReScore(contR2score, dictLay, R2scores, sortR2, r2_ii, runs):

    for i in np.linspace(0, dictLay['1']*(len(dictLay)-1), len(dictLay), dtype=int):
        for j in range(dictLay['1']):
            #contR2score.append((f'{tempRMSE[i+j]:.3e} | ').center(100,' ')+'\n')
            contR2score.append((f'{R2scores[i+j]:.3f}').center(9,' ')+' | ')
        contR2score.append('\n')

    contR2score.append('\n'+f' Three best R2score: '.center(100,'-')+'\n')
    np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
    contR2score.append((f'{array(sortR2)[:3]} - idx: {r2_ii[:3]} ').center(100,' ')+'\n')

    contR2score.append('\n'+(f' R2score cluster for three best RMSE: ').center(100,'-')+'\n')
    np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
    contR2score.append((f'{array(R2scores)[r2_ii[:3]]} - idx: {r2_ii[:3]} ').center(100,' ')+'\n')

    contR2score.append('\n'+f' Worse R2score: '.center(100,'-')+'\n')
    np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
    contR2score.append((f'{sortR2[3]:.3f} - idx: {r2_ii[3]} ').center(100,' ')+'\n')

    contR2score.append('\n'+(f' R2score cluster for Worse RMSE: ').center(100,'-')+'\n')
    contR2score.append((f'{array(R2scores)[r2_ii[-1]]:.3f} - idx: {r2_ii[-1]} ').center(100,' ')+'\n')

    contR2score.append('\n'+f" RMSE for each training ".center(100,"*")+'\n')
    
    return contR2score

##=========================================================== 
def wEndContReScore(contR2score, sortRMSE, RMSE, r2_ii, rmse_ii, runs_list):    

    contR2score.append('\n'+(f' Three best RMSE: ').center(100,'-')+'\n')
    for i in range(3):
        contR2score.append('\n'+(f' {runs_list[rmse_ii[i]]} ').center(100,'-')+'\n')
    
    np.set_printoptions(formatter={'float': '{: 0.3e}'.format})
    contR2score.append((f'{sortRMSE[:3]} - idx: {rmse_ii[:3]} ').center(100,' ')+'\n')

    contR2score.append('\n'+f' RMSE cluster for three best R2score: '.center(100,'-')+'\n')
    np.set_printoptions(formatter={'float': '{: 0.3e}'.format})
    contR2score.append((f'{RMSE[r2_ii[:3]]} - idx: {r2_ii[:3]} ').center(100,' ')+'\n')

    contR2score.append('\n'+(f' Worse RMSE: ').center(100,'-')+'\n')
    contR2score.append('\n'+(f' {runs_list[rmse_ii[-1]]} ').center(100,'-')+'\n')
    contR2score.append((f'{array(sortRMSE)[-1]:.3e} - idx: {r2_ii[-1]} ').center(100,' ')+'\n')

    contR2score.append('\n'+(f' RMSE cluster for Worse R2score: ').center(100,'-')+'\n')
    contR2score.append((f'{RMSE[r2_ii[-1]]:.3e} - idx: {r2_ii[-1]} ').center(100,' ')+'\n')

    contR2score.append('\n'+(f' Sorted Trainings for RMSE from best to worse: ').center(100,'-')+'\n')
    for i in rmse_ii:
        contR2score.append('\n'+(f' {runs_list[i]} ').center(70,' ')+f'| {sortRMSE[i]:.4f}'.center(10,' ')+'\n')
    
    return contR2score

##===========================================================
def wContActFunc(contActFunc, rmse_ii, runs_list, pathModel, r2score, InputDim, OutputDim, dictmodel, dictMetrics):

    for i in rmse_ii:
        run = runs_list[i]
        model     = None
        #model     = models.load_model(f'{pathModel}/model',custom_objects={'r2score':r2score})
        model     = models.load_model(f'{pathModel}/model',custom_objects=dictMetrics)
        dictmodel = model.get_config()

        nNeuron = run[run.find('Lay')+4:run.find('_Out')]
        version = run[run.find('_V')+2:run.find('s')+1]
        #neurons = list(eval(str(InputDim)+','+str(nNeuron).replace('_',',')+','+str(OutputDim)))
        try:
            neurons = list(eval(str(nNeuron).replace('_',',')))
        except:
            #neurons = list(nNeuron[0])
            neurons = list(eval(str(nNeuron).replace('_',',')+','+str(OutputDim)))
            #neurons = array(nNeuron,dtype=int)
            neurons = neurons[0]
        layers = len(dictmodel['layers'])-1    
        contActFunc.append(f' {run} '.center(100,'-')+'\n')
        try:
            contActFunc.append((f' Layer {1} - Neurons:'+ f'{InputDim}'.center(5,' ') +f' | Activation: {dictmodel["layers"][1]["config"]["activation"]["class_name"]} '.center(25,' ')).center(100,'-')+'\n')        
        except: 
            contActFunc.append((f' Layer {1} - Neurons:'+ f'{InputDim}'.center(5,' ') +f' | Activation: {dictmodel["layers"][1]["config"]["activation"]} '.center(25,' ')).center(100,'-')+'\n')

    if sys.getsizeof(neurons)==28:
        try:
            contActFunc.append((f' Layer 2 - Neurons:'+ f'{neurons}'.center(5,' ') +f' | Activation: {dictmodel["layers"][2]["config"]["activation"]["class_name"]} '.center(25,' ')).center(100,'-')+'\n')
        except:
            contActFunc.append((f' Layer 2 - Neurons:'+ f'{neurons}'.center(5,' ') +f' | Activation: {dictmodel["layers"][2]["config"]["activation"]} '.center(25,' ')).center(100,'-')+'\n')
    elif len(neurons) == 2:
        contActFunc.append((f' Layer {len(neurons)+1} - Neurons:'+ f'{neurons[0]}'.center(5,' ') +f' | Activation: {dictmodel["layers"][len(neurons)+1]["config"]["activation"]["class_name"]} '.center(25,' ')).center(100,'-')+'\n')
        contActFunc.append((f' Layer {len(neurons)+2} - Neurons:'+ f'{neurons[1]}'.center(5,' ') +f' | Activation: {dictmodel["layers"][len(neurons)+2]["config"]["activation"]["class_name"]} '.center(25,' ')).center(100,'-')+'\n')
    else :        
        for idx in range(2,layers):
            try:
                contActFunc.append((f' Layer {idx} - Neurons:'+ f'{neurons[idx-2]}'.center(5,' ') +f' | Activation: {dictmodel["layers"][idx]["config"]["activation"]["class_name"]} '.center(25,' ')).center(100,'-')+'\n')
            except:
                contActFunc.append((f' Layer {idx} - Neurons:'+ f'{neurons[idx-2]}'.center(5,' ') +f' | Activation: {dictmodel["layers"][idx]["config"]["activation"]} '.center(25,' ')).center(100,'-')+'\n')

        contActFunc.append((f' Layer {layers} - Neurons:'+ f'{OutputDim}'.center(5,' ') +f' | Activation: '+f'{dictmodel["layers"][layers]["config"]["activation"]} '.center(10,' ')).center(100,'-')+'\n\n')    

    return contActFunc, neurons

## ==============================================
def writeSigm_(sigmSign, dictData, nSpaces, rmsSpaces, of, pred, targ, signal, eLabel, cteAdjust=1 ):
    
    rmse_of, rmse_pred, sig_of, sig_pred, sig_targ = [], [], [], [], []
    if signal.lower()=='energy':
        #targ, of, pred = targ/1000, of/1000, pred/1000
        sigmSign.append( f'{eLabel}'.center(10)+' & '+f'{ np.mean( np.sum( targ, axis=1 ) ):.2f} $\pm$ { np.std( np.sum( targ, axis=1)):.2f}'.center(nSpaces) +' & '+ f'{getSigmClus( targ, targ ):.2f}'.center(rmsSpaces) + ' & ' +
                         f'{ np.mean( np.sum( of, axis=1 ) ):.2f} $\pm$ { np.std( np.sum( of, axis=1)):.2f}'.center(nSpaces) + '&'+ f'{rms_e( of.sum(axis=1), targ.sum(axis=1) ):.4f}'.center(rmsSpaces)+' & '+ f'{getSigmClus( of, targ ):.2f}'.center(rmsSpaces) + ' & '+
                         f'{ np.mean( np.sum( pred, axis=1 ) ):.2f} $\pm$ { np.std( np.sum( pred, axis=1)):.2f}'.center(nSpaces) + '&'+ f'{rms_e( pred.sum(axis=1), targ.sum(axis=1)):.4f}'.center(rmsSpaces) +' & '+ f'{getSigmClus( pred, targ ):.2f}'.center(rmsSpaces) + '\\\n')
        
        #dictData.update({ signal: {'rmseOF': rms_e( of.sum(axis=1), targ.sum(axis=1)) ,'rmseMLP': rms_e( pred.sum(axis=1), targ.sum(axis=1)) , 'sigTarg': getSigmClus( targ, targ ),'sigOF': getSigmClus( of, targ ), 'sigMLP': getSigmClus( pred, targ ) } } )

        try:            
            rmse_of   = dictData[signal]['rmseOF']
            rmse_pred = dictData[signal]['rmsePred']            

            rmse_of.append(rms_e( of.sum(axis=1), targ.sum(axis=1)))
            rmse_pred.append(rms_e( pred.sum(axis=1), targ.sum(axis=1)))
            
            sig_targ = dictData[signal]['sigTarg']
            sig_pred = dictData[signal]['sigPred']
            sig_of   = dictData[signal]['sigOF']            
            
            sig_targ.append( getSigmClus( targ, targ ) )
            sig_pred.append( getSigmClus( pred, targ ) )
            sig_of.append( getSigmClus( of, targ ) )
                                    
        except :
            rmse_of.append(rms_e( of.sum(axis=1), targ.sum(axis=1)))
            rmse_pred.append(rms_e( pred.sum(axis=1), targ.sum(axis=1)))
            
            sig_of.append( getSigmClus( of, targ ) )
            sig_pred.append( getSigmClus( pred, targ ) )
            sig_targ.append( getSigmClus( targ, targ ) )

        
    elif signal.lower() == 'time':
        sigmSign.append( f'{eLabel}'.center(10)+' & '+f'{   np.mean( targ )*cteAdjust:.2f} $\pm$ { np.std( targ )*cteAdjust:.2f}'.center(nSpaces) +' & '+ f'{np.mean( np.std( targ, axis=1 ))*cteAdjust:.2f}'.center(rmsSpaces) + ' & ' +
                         f'{ np.mean( of )*cteAdjust:.2f} $\pm$ { np.std( of )*cteAdjust:.2f}'.center(nSpaces) + '&'+ f'{rms_e( of.sum(axis=1), targ.mean(axis=1) )*cteAdjust:.4f}'.center(rmsSpaces)+' & '+ f'{np.mean( np.std( of, axis=1 ))*cteAdjust:.2f}'.center(rmsSpaces) + ' & '+
                         f'{ np.mean( pred )*cteAdjust:.2f} $\pm$ { np.std( pred )*cteAdjust:.2f}'.center(nSpaces) + '&'+ f'{rms_e( pred.sum(axis=1), targ.mean(axis=1))*cteAdjust:.4f}'.center(rmsSpaces) +' & '+ f'{np.mean( np.std( pred, axis=1 ))*cteAdjust:.2f}'.center(rmsSpaces))
        try:            
            rmse_of   = dictData[signal]['rmseOF']
            rmse_pred = dictData[signal]['rmsePred']            

            rmse_of.append(rms_e( of.sum(axis=1), targ.sum(axis=1)))
            rmse_pred.append(rms_e( pred.sum(axis=1), targ.sum(axis=1)))
            
            sig_targ = dictData[signal]['sigTarg']
            sig_pred = dictData[signal]['sigPred']
            sig_of   = dictData[signal]['sigOF']            
            
            sig_targ.append(np.mean( np.std( targ, axis=1 )))
            sig_pred.append(np.mean( np.std( pred, axis=1 )))
            sig_of.append(np.mean( np.std( of, axis=1 )))
                                    
        except :
            rmse_of.append(rms_e( of.sum(axis=1), targ.sum(axis=1)))
            rmse_pred.append(rms_e( pred.sum(axis=1), targ.sum(axis=1)))
            
            sig_of.append(np.mean( np.std( of, axis=1 )))
            sig_pred.append(np.mean( np.std( pred, axis=1 )))
            sig_targ.append(np.mean( np.std( targ, axis=1 )))
        
        #dictData.update({ signal: {'rmseOF': rmse_of ,'rmsePred': rmse_pred , 'sigTarg': sig_targ,'sigOF': sig_of, 'sigPred': sig_pred } } )
    else:
        print('Error! You must choose between ENERGY and TIME')
        sys.exit()    
    
    dictData.update({ signal: {'rmseOF': rmse_of ,'rmsePred': rmse_pred , 'sigTarg': sig_targ,'sigOF': sig_of, 'sigPred': sig_pred } } )
    
    return sigmSign, dictData

## ===================================================
def writeSigm(sigmSign, dictData, nSpaces, rmsSpaces, pred, of, targ, signal, eLabel, cteAdjust=1 ):
    
    rmse_of, rmse_pred, sig_of, sig_pred, sig_targ = [], [], [], [], []
    if signal.lower()=='energy':
        #targ, of, pred = targ/1000, of/1000, pred/1000
        sigmSign.append( f'{eLabel}'.center(10)+' & '+f'{ np.mean( np.sum( targ, axis=1 ) ):.2f} $\pm$ { np.std( np.sum( targ, axis=1)):.2f}'.center(nSpaces) +' & '+ f'{getSigmClus( targ, targ ):.4f}'.center(rmsSpaces) + ' & ' +
                         f'{ np.mean( np.sum( of, axis=1 ) ):.2f} $\pm$ { np.std( np.sum( of, axis=1)):.2f}'.center(nSpaces) + '&'+ f'{rms_e( of.sum(axis=1), targ.sum(axis=1) ):.4f}'.center(rmsSpaces)+' & '+ f'{getSigmClus( of, targ ):.4f}'.center(rmsSpaces) + ' & '+
                         f'{ np.mean( np.sum( pred, axis=1 ) ):.2f} $\pm$ { np.std( np.sum( pred, axis=1)):.2f}'.center(nSpaces) + '&'+ f'{rms_e( pred.sum(axis=1), targ.sum(axis=1)):.4f}'.center(rmsSpaces) +' & '+ f'{getSigmClus( pred, targ ):.4f}'.center(rmsSpaces))
        
    elif signal.lower()=='time':
        sigmSign.append( f'{eLabel}'.center(10)+' & '+f'{   np.mean( targ )*cteAdjust:.2f} $\pm$ { np.std( targ )*cteAdjust:.2f}'.center(nSpaces) +' & '+ f'{np.mean( np.std( targ, axis=1 ))*cteAdjust:.2f}'.center(rmsSpaces) + ' & ' +
                         f'{ np.mean( of )*cteAdjust:.2f} $\pm$ { np.std( of )*cteAdjust:.2f}'.center(nSpaces) + '&'+ f'{rms_e( of.sum(axis=1), targ.mean(axis=1) )*cteAdjust:.4f}'.center(rmsSpaces)+' & '+ f'{np.mean( np.std( of, axis=1 ))*cteAdjust:.2f}'.center(rmsSpaces) + ' & '+
                         f'{ np.mean( pred )*cteAdjust:.2f} $\pm$ { np.std( pred )*cteAdjust:.2f}'.center(nSpaces) + '&'+ f'{rms_e( pred.sum(axis=1), targ.mean(axis=1))*cteAdjust:.4f}'.center(rmsSpaces) +' & '+ f'{np.mean( np.std( pred, axis=1 ))*cteAdjust:.2f}'.center(rmsSpaces))

        #dictData.update({ signal: {'rmseOF': rms_e( of.sum(axis=1), targ.sum(axis=1)) ,'rmseMLP': rms_e( pred.sum(axis=1), targ.sum(axis=1)) , 'sigTarg': getSigmClus( targ, targ ),'sigOF': getSigmClus( of, targ ), 'sigMLP': getSigmClus( pred, targ ) } } )

        try:            
            rmse_of   = dictData[signal]['rmseOF']
            rmse_pred = dictData[signal]['rmsePred']            

            rmse_of.append(rms_e( of.sum(axis=1), targ.sum(axis=1)))
            rmse_pred.append(rms_e( pred.sum(axis=1), targ.sum(axis=1)))
            
            sig_targ = dictData[signal]['sigTarg']
            sig_pred = dictData[signal]['sigPred']
            sig_of   = dictData[signal]['sigOF']            
            
            sig_targ.append( getSigmClus( targ, targ ) )
            sig_pred.append( getSigmClus( pred, targ ) )
            sig_of.append( getSigmClus( of, targ ) )
                                    
        except :
            rmse_of.append(rms_e( of.sum(axis=1), targ.sum(axis=1)))
            rmse_pred.append(rms_e( pred.sum(axis=1), targ.sum(axis=1)))
            
            sig_of.append( getSigmClus( of, targ ) )
            sig_pred.append( getSigmClus( pred, targ ) )
            sig_targ.append( getSigmClus( targ, targ ) )
                
        #dictData.update({ signal: {'rmseOF': rmse_of ,'rmsePred': rmse_pred , 'sigTarg': sig_targ,'sigOF': sig_of, 'sigPred': sig_pred } } )
    else:
        print('Error! You must choose between ENERGY and TIME')
        sys.exit()    
    
    dictData.update({ signal: {'rmseOF': rmse_of ,'rmsePred': rmse_pred , 'sigTarg': sig_targ,'sigOF': sig_of, 'sigPred': sig_pred } } )
    
    return sigmSign, dictData
##########################################

def XTalk(x):
    """
    From C++:
    Xtalk ->SetParameter(0,g_Cx);
    Xtalk ->SetParameter(1,g_Rf);
    Xtalk ->SetParameter(2,g_Rin);
    Xtalk ->SetParameter(3,g_taud);
    Xtalk ->SetParameter(4,g_taupa);
    Xtalk ->SetParameter(5,g_td);    
    Xtalk ->SetParameter(6,g_ToNormXtC); 
    """ 
    g_ToNormXtC =   0.02242
    #g_ToNormXtC = 1
    g_taud      =  15.82 
    g_taupa     =  17.31 
    g_td        = 420.00 
    g_Rf        =   0.078     
    g_Rin       =   1.20 
    g_Cx        =  47.00
    par = np.zeros(7)
    par[0], par[1], par[2], par[3], par[4], par[5], par[6] = g_Cx, g_Rf, g_Rin, g_taud, g_taupa, g_td, g_ToNormXtC

    xt = (1/par[6]*((par[0]*par[1]*par[2]*(2*exp(x/par[3])*pow(par[3],2)*(x*(par[3] - par[4])*(par[4] + par[5]) + par[4]*(3*par[3]*par[4] + (2*par[3] + par[4])*par[5]))- exp(x/par[4]) * (pow(x,2)*pow(par[3] - par[4],2)*(par[3] + par[5]) - 2*x*par[3]*(par[3] - par[4])*(2*par[3]*par[4] + (par[3] + par[4])*par[5]) +  2*pow(par[3],2)*par[4]*(3*par[3]*par[4] + (2*par[3] + par[4])*par[5])) + par[3]*(-2*exp(x/par[3] + par[5]/par[4])*par[3]*par[4]*(x*(par[3] - par[4]) + 3*par[3]*par[4] + (-par[3] + par[4])*par[5]) +  exp(x/par[4] + par[5]/par[3]) * (pow(x,2)*pow(par[3] - par[4],2) + 6*pow(par[3],2)*pow(par[4],2) +  4*par[3]*(par[3] - par[4])*par[4]*par[5] + pow(par[3] - par[4],2)*pow(par[5],2) -  2*x*(par[3] - par[4]) * (2*par[3] *par[4] + (par[3] - par[4])*par[5])))*heaviside(x - par[5],1))))/(2.*exp(x*(1/par[3] + 1/par[4]))*par[3]*pow(par[3]- par[4],4)*par[5]))    
    
    return xt


## ==========================================================

idx_7x7 = array(range(49))
idx_5x5 = list( getIdxClus_mxn(idx_7x7.reshape(7,7), 5, 5) )
idx_3x3 = list( getIdxClus_mxn(idx_7x7.reshape(7,7), 3, 3) )
ij_cell = ['-3,3' , '-2,3' , '-1,3' , '0,3' , '1,3' , '2,3' , '3,3' , 
           '-3,2' , '-2,2' , '-1,2' , '0,2' , '1,2' , '2,2' , '3,2' , 
           '-3,1' , '-2,1' , '-1,1' , '0,1' , '1,1' , '2,1' , '3,1' , 
           '-3,0' , '-2,0' , '-1,0' , '0,0' , '1,0' , '2,0' , '3,0' , 
           '-3,-1', '-2,-1', '-1,-1', '0,-1', '1,-1', '2,-1', '3,-1', 
           '-3,-2', '-2,-2', '-1,-2', '0,-2', '1,-2', '2,-2', '3,-2', 
           '-3,-3', '-2,-3', '-1,-3', '0,-3', '1,-3', '2,-3', '3,-3' ]
