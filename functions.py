#!/usr/bin/env python
# coding: utf-8

## This functions are for studies in crosstalk mitigation on LAr calorimeter cells
## of the ATLAS experiment
##
## Author: Marton S Santos
## 
"""
This file is part of PH.D thesis to crosstalk mitigation using ML methods.

This code is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by the
Free Software Foundation, either version 3 of the License, or (at your option)
any later version.

This code is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for
more details.

You should have received a copy of the GNU General Public License along with
this program. If not, see <https://www.gnu.org/licenses/>.

Copyright (C) 2024 Marton S Santos.
"""

### ========================================================
# Declare which functions will be exposed when using `from script import *`
__all__ = [
    'autocorr', 'cellFunction', 'createPath', 'derivCellFunction', 
    'derivXtalkFunction', 'DateInfo', 'genCellSamples', 'genXTcSamples', 'genXTlSamples', 
    'genSampDelay', 'genNoise', 'getIdxClus_mxn', 'getIdxSampClus', 
    'loadSaveDict', 'min_max', 'OptFilt', 'plotHeatmap', 
    'plotSS', 'plotHisto', 'plotScatter', 'readDirectCells', 'plotSigmClus', 'rms',
    'timeInfo', 'XTalk'
]

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

import time
from XTconstants import *


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
    """
        get the time stamp 
    """
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
def genCellSamples(vectDelay, g_tau_0, nSamp):
    #ClustCellSamples, derivCellSamples = np.zeros(vectDelay.shape*nSamp), np.zeros(vectDelay.shape*nSamp)
    clustCellSamples, derivCellSamples = [], []
    for cellDelay in vectDelay:
        for n in range(nSamp):
            clustCellSamples.append( cellFunction( 25*(n+1) + cellDelay + g_tau_0 ))
    
    return array(clustCellSamples)

##===========================================================
def loadSaveDict(filename, **kwargs):
    dataDict = kwargs.get('dataDict')
    load = kwargs.get('load')
    save = kwargs.get('save')
    
    if load is not None:
        if os.path.exists(filename):  # Check if the file exists
            with open(filename, "rb") as file:
                loadedDict = pkl.load(file)
            if loadedDict is None or (isinstance(loadedDict, (dict, list, np.ndarray)) and len(loadedDict) == 0):
                print("The file is empty.")
            else:
                return loadedDict
        else:
            print("The file does not exist.")
            return 0  # Handle the case where the file doesn't exist
        
    if save is not None:
        with open(filename, "wb") as file:
            pkl.dump(dataDict, file)

## =========================================================
## =========================================================
def getMeanRms(x1, MeanRms='mean', decimals=2):
    N = np.shape(x1)[0]
    # Determine whether to calculate 'mean' or 'rms'
    if MeanRms.lower() == 'mean':
        means = np.maximum(np.round(np.mean(x1, axis=1), decimals), 10**(-decimals))  # Calculate mean for each column
        ## Calculate the std error rounded to decimals precision, and fix to 0.01 if 0.0 as a result
        stdError = np.maximum(np.round(np.std(x1, axis=1)/np.sqrt(N), decimals), 10**(-decimals) )
        
    elif MeanRms.lower() == 'rms':
        means = np.round( rms(x1, axis=1), decimals )      # Calculate rms for each column
        #stds = np.std(x1, axis=1)    # Calculate std for each column
        stdError = np.maximum(np.round(np.std(x1, axis=1)/np.sqrt(N), decimals), 10**(-decimals) )    # Calculate std for each column
        
    else:
        raise ValueError("MeanRms should be either 'mean' or 'rms'.")

    # Intercalate means and stds dynamically
    results = []
    for mean, stdErr in zip(means, stdError):
        results.append([mean, stdErr])
        #results.append(std)

    return results

## =======================================
def getIdxClus_mxn(cluster, m, n):
    """
        return the cell index.
        cluster
        m - actual order
        n -  desired new order
    """
    #m, n = 5,5
    cluster = cluster.reshape(m,m)
    row, col = np.shape(cluster)[0] // 2, np.shape(cluster)[1] // 2    
    idx_mxn = cluster[row - n // 2:row + n // 2 + 1, col - n // 2:col + n // 2 + 1]
    
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

## =============================================
def genSampDelay(self, n=None):
    tDelay = []
    if n is None: n=49 ## for a 7x7 cluster celss

    while len( tDelay )<n:
        data = np.random.normal(0, 0.1)
        if abs(data) > 0.1: pass
        else: tDelay.append( abs(data) ) 

    return np.array( tDelay )
## ==========================================================

def genXTcSamples(vectDelay, g_tau_0, nSamp):    
    #g_AmpXt_C   = 4.0/100         ## XTalk amplitude on % values of the Energy
    ClustXTcSamples = []
    for cellDelay in vectDelay:
        for n in range(nSamp):        
            ClustXTcSamples.append( g_AmpXt_C*XTalk( 25*( n+1 ) + cellDelay + g_tau_0 ))

    return array(ClustXTcSamples)

## ==========================================================

def genXTlSamples(vectDelay, g_tau_0, nSamp):    
    g_AmpXt_L   = 2.3/100         ## XTalk amplitude on % values of the Energy
    ClustXTlSamples = []
    for cellDelay in vectDelay:
        for n in range(nSamp):                   
            ClustXTlSamples.append( g_AmpXt_L*XTalk( 25*(n+1) + cellDelay + g_tau_0 ))
    ''
    return array(ClustXTlSamples)

## ==========================================================
def genNoise(n=None, norm=False):    
    g_AmpNoise    = 50   ## MeV    
    noise = np.zeros(n)
    for i in range(n):
        noise[i] = np.random.normal(0,2)

    return array(g_AmpNoise*(noise/max(noise)))
## ==========================================================
def genRandVal(count, start, end):
    values = set()
    while len(values) < count:
        value = random.randint(start, end)
        values.add(value)
    return list(sorted(values, reverse=True))
    
##===========================================================
## Optinal Filtering (Baseline)
def OptFilt(samples, ai=None, bi=None):
    """
       Computes the Amplitude and time of flights for LAr cells samples
    """
    idx_7x7 = array(range(49))
    idx_5x5 = getIdxClus_mxn(idx_7x7, 7, 5 )
    idx_3x3 = getIdxClus_mxn(idx_7x7, 7, 3 )
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
def plotHeatmap(data, **kwargs):
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
            xLabel = r'Energy [$10^{-3}$'+f' GeV]'
        else :
            xLabel = f'Energy [{params["unit"]}]'
    elif params["label"].lower() == 'time':        
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

### ============================= 
def plotScatter(yPred, yTrue, **kwargs):
    #fig = plt.figure()    
    default_params = {
        "ext": "png",
        "fileName": "",
        "label": "",
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
    ax[0].scatter(yTrue, yPred, s=7, label=f'{params["label"]}')
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


## =========================================================
def min_max(x1):
    """
        x1 is list of arrays
    """
    return np.min(x1), np.max(x1)

##===========================================================
def rms(x, axis=None):
    return np.sqrt(np.mean(x**2, axis=axis))  

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

