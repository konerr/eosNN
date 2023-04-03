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

####################Functions for use in 2-Species, IG + JWL PTE Data collecting script

def LH_SampleSpace(npts,nvars,lbound,ubound):
#Function which returns a Latin hypercube sampling consisting of npts within the space
#with bounds given by the nvars-dimensional vectors lbound and ubound

	if (len(lbound) != len(ubound)):
		print('Error: Lower and Upper bound vectors are not the same size!') 
		raise Exception("Quitting code...")
	if (len(lbound) != nvars):
		print('Error: Lower and Upper bound vectors have differing dimensions than the sampling space!')
		raise Exception("Quitting code...")

	sampler = sp.stats.qmc.LatinHypercube(d=nvars)
	sample  = sampler.random(n=npts)
	sspace  = sp.stats.qmc.scale(sample,lbound,ubound) 

	return sspace

def EvalIGEos(rho,e,g,cv):
	P = (g-1.0)*rho*e
	T = e/cv
	a = (g*(g-1.0)*e)**0.5
	return P, T, a

def EvalJWLEos(rho,e,A,B,R1,R2,rho0,omega,cv):
	drat = rho0/rho
	exp1 = np.exp(-1.0*R1*drat)
	exp2 = np.exp(-1.0*R2*drat)
	om1  = omega*(1.0+omega)/R1/rho0
	om2  = omega*(1.0+omega)/R2/rho0

	P  = A*(1.0-omega/R1/drat)*exp1 + B*(1.0-omega/R2/drat)*exp2 + omega*rho*e
#	T  = (1.0/omega/rho/cv)*(P-A*exp1-B*exp2)
	T  = (1.0/cv)*(e - (A/R1/rho0)*exp1 - (B/R2/rho0)*exp2) 
	a2 = (A*exp1)*(R1*drat-om1) + (B*exp2)*(R2*drat-om2)+omega*(1.0+omega)*e
	a  = a2**0.5 

	return P, T, a

def RFLU_InvertMatrix(A,n):
#Note: This function didn't port over very cleanly from Rocflu...will use numpy's
#      built-in inverter for now...
	ir = np.arange(0,n,1)
	jr = np.arange(0,2*n,1)
	
	augmat = np.zeros((n,2*n))

	for i in np.arange(0,n,1):
		for j in np.arange(0,2*n,1):
			if (j <= n-1):
				augmat[i][j] = A[i][j]
			elif (i+n == j):
				augmat[i][j] = 1.0
			else:
				augmat[i][j] = 0.0

	for k in np.arange(0,n-1,1):
		for j in np.arange(k,n,1):
			if (augmat[k][k] == 0):
				for i in np.arange(k,n,1):
					if (augmat[i][k] != 0.0):
						for l in np.arange(1,2*n,1):
							augmat[k][l] = augmat[k][l] + augmat[i][l]

	for k in np.arange(0,n-1,1):
		for j in np.arange(k,n,1):
			m = augmat[j][k]/augmat[k][k]
			for i in np.arange(k-1,2*n,1):
				augmat[j][i] = augmat[j][i] - m*augmat[k][i]

	for i in np.arange(0,n,1):
		if (augmat[i][i] == 0):
			print('Error: Matrix is non-invertible!')
			raise Exception('Quitting code...')

	for i in np.arange(0,n,1):
		m = augmat[i][i]
		for j in np.arange(i-1,2*n,1):
			augmat[i][j] = augmat[i][j]/m

	for k in np.arange(n-2,-1,-1):
		for i in np.arange(0,k,1):
			m = augmat[i][k+1]
			for j in np.arange(k-1,2*n,1):
				augmat[i][j] = augmat[i][j] - augmat[k+1][j]*m

	for i in np.arange(0,n,1):
		for j in np.arange(0,n,1):
			Ainv[i][j] = augmat[i][j+n]		
	
	return Ainv

def BroydenInvSolve(g,cvi,A,B,R1,R2,r0,w,cve,e,r,Y):
#Function which solves the 2-species PTE system using Broyden's inverse method
#Two different initial guesses are used to help the stability of the solver w.r.t the specific volume guess
	niter= 100
	tol  = 1.0e-10
	tolok= 1.0e-3 
	fuzz = 1.0e-15 #value to defend against division by zero later

	sol  = np.zeros(4)
	sol2 = np.zeros(4)
	f    = np.zeros(4)
	f2   = np.zeros(4)
	Ja   = np.zeros((4,4))
	Ja2  = np.zeros((4,4))
	v    = 1.0/r

	sol[0] = v/(2.0*(1.0-Y))
	sol[1] = v/(2.0*Y)
	sol[2] = e
	sol[3] = e 

	sol2[0]   = sol[1]
	sol2[1]   = sol[0]
	sol2[2:3] = sol[2:3]

	sol_old  = sol
	sol_old2 = sol2	

	va = sol[0]
	vb = sol[1]
	ea = sol[2]
	eb = sol[3]
	va2 = sol2[0]
	vb2 = sol2[1]
	ea2 = sol2[2]
	eb2 = sol2[3]

	f[0] = va*Y + vb*(1.0-Y) - v
	f[1] = ea*Y + eb*(1.0-Y) - e
	ra   = 1.0/va
	rb   = 1.0/vb
	[Pb,Tb,dum]   = EvalIGEos(rb,eb,g,cvi)
	[Pa,Ta,dum]   = EvalJWLEos(ra,ea,A,B,R1,R2,r0,w,cve)
	f[2] = Pa-Pb
	f[3] = Ta-Tb

	f2[0] = va2*Y + vb2*(1.0-Y) - v
	f2[1] = ea2*Y + eb2*(1.0-Y) - e
	ra2   = 1.0/va2
	rb2   = 1.0/vb2
	[Pb2,Tb2,dum]   = EvalIGEos(rb2,eb2,g,cvi)
	[Pa2,Ta2,dum]   = EvalJWLEos(ra2,ea2,A,B,R1,R2,r0,w,cve)
	f2[2] = Pa2-Pb2
	f2[3] = Ta2-Tb2

	Ja[0][0]   = Y
	Ja[0][1]   = 1.0-Y
	Ja[0][2:3] = 0.0	
	Ja[1][0:1] = 0.0
	Ja[1][2]   = Y
	Ja[1][3]   = 1.0-Y

	dfdva = A*(w/va - R1*r0 + w/R1/r0/va**2.0)*np.exp(-1.0*R1*r0*va) + B*(w/va - R2*r0 + w/R2/r0/va**2.0)*np.exp(-1.0*R2*r0*va)
	Ja[2][0] = -1.0*w*ea/va**2.0 + dfdva
	Ja[2][1] = (g-1.0)*eb/vb/vb
	Ja[2][2] = w/va
	Ja[2][3] = -1.0*(g-1.0)/vb

	dgdva = A/cve*np.exp(-1.0*R1*r0*va)+ B/cve*np.exp(-1.0*R2*r0*va)
	Ja[3][0] = dgdva
	Ja[3][1] = 0.0
	Ja[3][2] = 1.0/cve
	Ja[3][3] = -1.0/cvi #(1.0-g)/gc

	Ja2[0][0]   = Y
	Ja2[0][1]   = 1.0-Y
	Ja2[0][2:3] = 0.0
	Ja2[1][0:1] = 0.0
	Ja2[1][2]   = Y
	Ja2[1][3]   = 1.0-Y

	dfdva = A*(w/va2 - R1*r0 + w/R1/r0/va2**2.0)*np.exp(-1.0*R1*r0*va2) + B*(w/va2 - R2*r0 + w/R2/r0/va2**2.0)*np.exp(-1.0*R2*r0*va2)
	Ja2[2][0] = -1.0*w*ea2/va2**2.0 + dfdva
	Ja2[2][1] = (g-1.0)*eb2/vb2/vb2
	Ja2[2][2] = w/va2
	Ja2[2][3] = -1.0*(g-1.0)/vb2

	dgdva = A/cve*np.exp(-1.0*R1*r0*va2)+ B/cve*np.exp(-1.0*R2*r0*va2)
	Ja2[3][0] = dgdva
	Ja2[3][1] = 0.0
	Ja2[3][2] = 1.0/cve
	Ja2[3][3] = -1.0/cvi #(1.0-g)/gc

	InvJa = np.linalg.inv(Ja)
	InvJa2= np.linalg.inv(Ja2)

	alt = np.zeros(4)
	alt2= np.zeros(4)

	for i in np.arange(0,4,1):
		for j in np.arange(0,4,1):
			alt[i] = alt[i] + InvJa[i][j]*f[j]		
			alt2[i]= alt2[i] + InvJa2[i][j]*f2[j]
	sol = sol_old - alt
	sol2 = sol_old2 - alt2

	for it in np.arange(0,niter,1):
		f_old = f
		va = sol[0]
		vb = sol[1]
		ea = sol[2]
		eb = sol[3]

		f_old2 = f2
		va2 = sol2[0]
		vb2 = sol2[1]
		ea2 = sol2[2]
		eb2 = sol2[3]

		f[0] = va*Y + vb*(1.0-Y) - v
		f[1] = ea*Y + eb*(1.0-Y) - e
		ra   = 1.0/va
		rb   = 1.0/vb
		[Pb,Tb,dum]   = EvalIGEos(rb,eb,g,cvi)
		[Pa,Ta,dum]   = EvalJWLEos(ra,ea,A,B,R1,R2,r0,w,cve)
		f[2] = Pa-Pb
		f[3] = Ta-Tb

		difff = f-f_old #y
		diffsol = sol-sol_old #s

		f2[0] = va2*Y + vb2*(1.0-Y) - v
		f2[1] = ea2*Y + eb2*(1.0-Y) - e
		ra2   = 1.0/va2
		rb2   = 1.0/vb2
		[Pb,Tb,dum]   = EvalIGEos(rb2,eb2,g,cvi)
		[Pa,Ta,dum]   = EvalJWLEos(ra2,ea2,A,B,R1,R2,r0,w,cve)
		f2[2] = Pa-Pb
		f2[3] = Ta-Tb

		difff2 = f2-f_old2 #y
		diffsol2 = sol2-sol_old2 #s

		difff_n = difff[0]**2.0 + difff[1]**2.0 + difff[2]**2.0 + difff[3]**2.0 #yTy
		difff_n2 = difff2[0]**2.0 + difff2[1]**2.0 + difff2[2]**2.0 + difff2[3]**2.0

		#addition
		temp1 = np.zeros(4)
		temp12= temp1
		res   = np.zeros((4,4))
		res2  = res
		for i in np.arange(0,4,1):
			for j in np.arange(0,4,1):
				temp1[i] = temp1[i] + InvJa[i][j]*difff[j]
				temp12[i] = temp12[i] + InvJa2[i][j]*difff2[j]

		temp2  = diffsol  - temp1
		temp22 = diffsol2 - temp12

		for j in np.arange(0,4,1):
			for k in np.arange(0,4,1):
				res[j][k] = temp2[j]*difff[k]/(difff_n+fuzz)
				res2[j][k] = temp22[j]*difff2[k]/(difff_n2+fuzz)
                #Update Inverse Jacobian
		InvJa = InvJa + res
		InvJa2 = InvJa2 + res2	
		#reset solution for iteration
		sol_old = sol
		sol_old2= sol2

		for i in np.arange(0,4,1):
			for j in np.arange(0,4,1):
				alt[i] = alt[i] + InvJa[i][j]*f[j]
				alt2[i] = alt2[i] + InvJa2[i][j]*f2[j]
		#Update Solution
		sol = sol_old - alt
		sol2 = sol_old2 - alt2			

		sNorm = ((sol[0]-sol_old[0])**2.0+(sol[1]-sol_old[1])**2.0+(sol[2]-sol_old[2])**2.0+(sol[3]-sol_old[3])**2.0)
		sNorm2 = ((sol2[0]-sol_old2[0])**2.0+(sol2[1]-sol_old2[1])**2.0+(sol2[2]-sol_old2[2])**2.0+(sol2[3]-sol_old2[3])**2.0)

		isn = np.isnan(sNorm)
		isn2 = np.isnan(sNorm2)
		#Convergence check 
		if (sNorm < tol and isn == 0):
			va = sol[0]
			vb = sol[1]
			e_exp = sol[2]
			e_id  = sol[3]
			break
		elif (sNorm2 < tol and isn2 == 0):
			va = sol2[0]
			vb = sol2[1]
			e_exp = sol2[2]
			e_id  = sol2[3]
			break
		elif (it == niter-1): #reached max iterations, print warning and give current solution to routine
			print('Broyden solver reached max number of iterations! Given solution not converged, increase max number of iterations if a strictly converged solution is desired.')
			va = sol[0]
			vb = sol[1]
			e_exp = sol[2]
			e_id  = sol[3]
  

	r_exp = 1.0/va
	r_id  = 1.0/vb
	[P_id,T_id,a_id]      = EvalIGEos(r_id,e_id,g,cvi)
	[P_exp,T_exp,a_exp]   = EvalJWLEos(r_exp,e_exp,A,B,R1,R2,r0,w,cve) 

	if ((P_id-P_exp)/P_exp < tolok and (T_id-T_exp)/T_exp < tolok):
		P = P_exp
		T = T_exp
		a = Y*a_exp + (1.0-Y)*a_id
	else:   #Solution not converged, warn user but still output (with 0 for SoS) to allow sampling to continue for other points
		print('Iterative solution not converged! Either increase acceptable tolerance or debug point!')
		print('rho,e,Y:',r,e,Y)
		P = min(P_exp,P_id)
		T = min(T_exp,T_id)
		a = 0.0   
	return P,T,a,r_exp,r_id,e_exp,e_id
