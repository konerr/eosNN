#!/usr/bin/python
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from scipy import optimize,stats
from scipy.stats import qmc
import random as rand
import math as m
#import sys
import csv
import os,glob
import sys


#####################################################################################
#Function Calls!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#####################################################################################

#Function calls from file
from PTE_2SpeciesFunctions import LH_SampleSpace 
from PTE_2SpeciesFunctions import EvalIGEos
from PTE_2SpeciesFunctions import EvalJWLEos
from PTE_2SpeciesFunctions import RFLU_InvertMatrix
from PTE_2SpeciesFunctions import BroydenInvSolve

#####################################################################################
#Main Script - Input
#####################################################################################
#Sampling Space
npts   = 150           #Number of wanted sampling points
rhomin = 1.0           #Minimum density of sampling space (kg/m^3)
rhomax = 1770.0        #Maximum density of sampling space
emin   = 1.25e5        #Minimum specific internal enery (sie) of sampling space (J/kg)
emax   = 1.5e7         #Maximum sie of sampling space

#EoS Parameters (currently set to air-PETN)
gamma = 1.4            #Ratio of specific heats for ideal gas
Cv_Id = 767.0          #Specific heat at constant volume (J/kg/K)

A     = 617.0e9        #High-pressure JWL coefficient (Pa)
B     = 16.926e9       #Medium-pressure JWL coefficient (Pa)
R1    = 4.4            #High-pressure JWL decay constant
R2    = 1.2            #Medium-pressure JWL decay constant
rho0  = 1770.0         #Initial unreacted HE density (kg/m^3)
omega = 0.25           #Gruniesen gamma for JWL EoS
Cv_JWL= 516.981        #Specific heat at constant volume (J/kg/K) 

#####################################################################################
#Samping Parameter Space
#####################################################################################
#Note: Units are assumed to be 'mks' unless explictly stated otherwise
#Note 2: Do not touch nspecies/nvars variables if using for a 2 species mixture!
nspecies = 2
nvars    = 2+(nspecies-1) 

lenpts = np.arange(0,npts,1)  #Enumerating the indices for the sampling points

mix_pts = LH_SampleSpace(npts,nvars,[rhomin,emin,0.0],[rhomax,emax,1.0])


#Init/allocate output and debug/sanity check arrays
P = np.zeros(npts)
T = np.zeros(npts)
a = np.zeros(npts)
r_exp = np.zeros(npts)
e_exp = r_exp
r_id  = r_exp
e_id  = r_exp
##################################################
#PTE Solve Routine Loop
##################################################

for n in lenpts:
	[P[n],T[n],a[n],r_exp[n],r_id[n],e_exp[n],e_id[n]] = BroydenInvSolve(gamma,Cv_Id,A,B,R1,R2,rho0,omega,Cv_JWL,mix_pts[n][1],mix_pts[n][0],mix_pts[n][2])
	

##################################################
#Output File Write
##################################################

#Write out inputs/outputs into a space-delimited data file
# with open('PTEdata.dat', 'w') as fileout:
# 	for i in lenpts: 
# 		if (P[i]>0 and T[i]>0):
# 			fileout.write(str(mix_pts[i][0])+' ')
# 			fileout.write(str(mix_pts[i][1])+' ')
# 			fileout.write(str(mix_pts[i][2])+' ')
# 			fileout.write(str(P[i])+' ')
# 			fileout.write(str(T[i])+' ')
# #			fileout.write(str(a[i])+' ')
# 			fileout.write('\n')

mask = (P>0) & (T>0) # Filter-out -ves and nans
data = np.concatenate((mix_pts[mask],np.array([P[mask],T[mask]]).T),axis=1)
hdr = 'rho, e, Y, P, T'
np.savetxt('PTEdata.dat', data, header=hdr )
np.savez_compressed('PTEdata', input=mix_pts[mask], output=np.array([P[mask],T[mask]]).T)

#Plot out desired plots
#plt.rcParams['text.usetex'] = True #Enable Latex for plots
#plt.plot()
#plt.axis()
#plt.xlabel()
#plt.ylabel()
#plt.show()
#plt.savefig('', dpi=None,orientation='portrait', format='png',
#        transparent=False, metadata=None)
#plt.close()
