from math import atan, sinh, cosh, tanh, exp, log, sqrt, sin, cos, gamma
import numpy as np
## Delay from each cell. Each cell has a dt on clock and this variation is constant between two cells. The sigma is 300 ps.
#pragma link C++ class vector<Double_t>+

## /********************************/
## Standard Cluster is defined on the Layer 2
## Standard Window. Data for Layer 2 (S2)
##


g_DeltaEtaS2 = 0.025         ## Delta Eta on S2
g_DeltaPhiS2 = np.pi/128.     ## Delta Phi on S2

g_nEtaS2     = 5            ## Cluster on layer 2 (Eta x Phi) - rows
g_nPhiS2     = 5            ## Cluster on layer 2 (Eta x Phi) - columns 


#g_tau0 = 0.5                ## arbitrary value for time bias

## Impact Point
g_Eta0 = 0.025  
g_Phi0 = np.pi/128 

## Impact point (center cluster) adjusted
g_EtaCenter = 0.4*g_Eta0 
g_PhiCenter = 0.4*g_Phi0 

g_randEta = np.random.uniform()
g_randPhi = np.random.uniform()

## Insert some fluctuations on impact point
g_EtaImpact = g_EtaCenter - g_DeltaEtaS2/2 + g_randEta*g_DeltaEtaS2 
g_PhiImpact = g_PhiCenter - g_DeltaPhiS2/2 + g_randPhi*g_DeltaPhiS2 

## *********************************

g_gainAdj     = 1             ##/ConstNormCluster[g_RowClust-1][g_ColClust-1]       ## adjust for normalization of 
g_E0          = 5e4           ## Impact Energy (in MeV)
NoiseAmp      =  50           ## amplitude of the noise on MeV
g_ADCpedestal = 900 
g_theta       = 2*atan(exp(-g_EtaImpact)) 
g_R0          = 1500  ##1385.0        ## distance between the beam axes and the first layer of calorimeter. Start point of EM Calorimeter (mm)
## /* *************************************** */ 
## Lenghts of the sampling layers
    ## g_Rs1 + g_Rs2 + g_Rs3 = 470
g_Rs1         =   90          ## length of the S1 in mm
g_Rs2         =  337          ## length of the S2 in mm
g_Rs3         =   43          ## length of the S3 in mm
## /* *************************************** */ 

g_tmax2   = 1000.   
g_dt      =    1.0            ## increment of time to generate graphical, in ns
g_window  =  600.0            ## size of window in ns
g_nPoints = g_window/g_dt     ## number of points for a window of 600 ns
g_tSamp   =   25.0            ## sample time in ns
g_nSamp   =    4.0            ## number of samples

g_ToNormXtC = 0.022206        ## This value normalize amplitude of the Xt_C to unit
##g_ToNormXtL = 0.0539463       ## This value normalize amplitude of the Xt_L to unit
g_ToNormXtL = 0.0539463       ## This value normalize amplitude of the Xt_L to unit
g_ToNormNoise = 4.4787   
##g_AmpXt_C   = 7.0/100         ## XTalk amplitude on % values of the Energy

g_AmpXt_C   = 4.0/100         ## Capacitive XTalk amplitude on % values of the Energy
g_AmpXt_L   = 2.3/100         ## Inductive XTalk amplitude on % values of the Energy
g_AmpXt_R   = 1.0/100         ## Resistive XTalk amplitude on % values of the Energy
##g_AmpNoise  = NoiseAmp/g_E0   ## Noise amplitude 
#g_AmpNoise  = 1./100   ## Noise amplitude 
g_AmpNoise  = 50   ## MeV

## /* *************************************** */
##/ parameters for the cell and Xtalk signals
g_taud    =   15.82 
g_taupa   =   17.31 
g_td      =  420.00 
g_Rf      =    0.078 
g_C1      =   50.00 
g_Rin     =    1.20 
g_Cx      =   47.00 
## /* *************************************** */

## /* *************************************** */
## Data for calculating Moliere Radius for a sampling calorimeter
## http:##pdg.lbl.gov/2019/AtomicNuclearProperties/index.html
g_da      =   4.00            ## thickness of the active media (Ar)
g_dp      =   2.00            ## thickness of the passive media (Pb)

g_RmLAr   =  90.43            ## Moliere Radius in mm
g_Z_LAr   =  18.00            ## Atomic number
g_X0_LAr  = 140.00            ## Radiation length in mm
g_wLAr    =   0.36            ## weight for the LAr
g_EcLAr   =  32.84            ## critical energy for e- on LAr in MeV, 31.91 MeV (for e+)

g_RmLead  =  16.02            ## Moliere Radius in mm
g_Z_Lead  =  82.00            ## Atomic number
g_X0_Lead =   5.612           ## Radiation length in mm
g_wLead   =   1 - g_wLAr      ## weight for the Lead
g_EcLead  =   7.43            ## critical energy for the Lead in MeV (e-), 7.16 for e+

g_Es      =  21.2             ## multiple scattering energy in MeV

##/ Those are alpha em beta parameters for the longitudinal profile
g_a       =   4.36            ## Adjusted accord to information of electron deposition on layers
g_b       =   0.25            ## Same as a 

##g_a       =   4.00 
##g_b       =   0.26 

##LAr resolution terms
g_SampTerm  = 10./100          ## 10% of the Energy
g_ConstTerm = 0.7/100          ## Constant term equal to 0.7%
g_NoiseTerm = 0.40             ## Energy on GeV. Noise term is equal to 400 MeV

## /* *************************************** */
## Effective values for a sampling calorimeter. Reference: ATL-COM-PHYS-2004-015

## Moliere Radius for a sampling calorimeter

g_Rmoleff = 1/(1/g_Es*(g_wLAr*g_EcLAr/g_X0_LAr + g_wLead*g_EcLead/g_X0_Lead)) 
g_X0eff   = 1/(g_wLAr/g_X0_LAr + g_wLead/g_X0_Lead) 
g_Eceff   = g_X0eff * ( (g_wLAr*g_EcLAr)/g_X0_LAr + (g_wLead*g_EcLead)/g_X0_Lead) 
g_Zeff    = g_wLAr*g_Z_LAr + g_wLead*g_Z_Lead 

e = 1/(1 + 0.007*(g_Z_Lead - g_Z_LAr))
T  = (g_a - 1)/g_b 

##/ Grindhammer parameters to fit shower

##/ Homogeneus media
z1 = 0.0251 + 0.00319*log(g_E0) 
z2 = 0.1162 - 0.000381*g_Z_LAr 
k1 = 0.659 - 0.00309*g_Z_LAr 
k2 = 0.645 
k3 = -2.59 
k4 = 0.3585 + 0.0421*log(g_E0) 
p1 = 2.632 - 0.00094*g_Z_LAr 
p2 = 0.401 + 0.00187*g_Z_LAr 
p3 = 1.313 - 0.0686*log(g_E0) 
y  = g_E0/g_Eceff 
t1hom = -0.59 
t1sam = -0.59 
Thom  = t1hom + log(y) 
t2 = -0.53 

##/ Sampling media
Fs = g_X0eff/(g_da + g_dp) 
Tsamp = (1 - e)*t2 + t1sam/Fs + Thom 