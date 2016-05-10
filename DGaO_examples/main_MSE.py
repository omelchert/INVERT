""" FILE:  main_MSE_EX2.py

Script performing exemplary solution of the Volterra kernel reconstruction
problem for the simplified optoacoustic problem on the beam-axis assuming a
Gaussian irradiation source profile, see Eq. (40) in section 1.3 "Diffraction
of laser-excited acoustic pulses" in the article

  Time-resolved laser optoacoustic tomography of inhomogeneous media,
  Karabutov, A.A and Podymova, N.B. and Letokhov, V.S., 
  Appl. Phys. B 63 (1996) 545-563

Author: O. Melchert
Date:   09.05.2016
"""
import sys; sys.path.append('../src/')
import os
import numpy as np
from OAVolterra_convolutionKernel import *
from scipy.optimize import fmin,minimize_scalar,minimize,root

def fetchSyntheticSignal(fName):
        x  = []
        f0 = []
        fx = []
        with open(fName,'r') as f:
                for line in f:
                   c = line.split()
                   if len(c)>1 and line[0]!='#':
                         x.append(float(c[1]))
                         f0.append(float(c[2]))
                         fx.append(float(c[3]))
                   if line[0]=='#':
                         if c[1] == 'wD':
                             wD = float(c[3])
                         if c[1] == 'D':
                             D = float(c[3])
        return np.asarray(x), np.asarray(f0), np.asarray(fx), wD, D 

def sourceReconstruction(pn,pz,wD,dt,N):
        cn  = 1.0
        nIt = 0
        while(cn>0.00000001):
                pn,cn = OAVolterra_inverse_PicardIteration(pn,pz,wD,dt,N)
                nIt += 1
        return pn,cn,nIt

def quadrature(f,t):
        N=len(f)
        fQuad = np.zeros(N)
        cumSum = 0.
        for i in range(1,N-1):
                cumSum += 0.25*(f[i-1]+f[i+1])*(t[i+1]-t[i-1])
                fQuad[i] += cumSum 
        return fQuad

def yNormCurve(p0,p,(z,zMin,zMax)):
        dz = z[1]-z[0]
        z1 = int((zMin-z[0])/dz)
        z2 = int((zMax-z[0])/dz)

        def myFunc(a):
            s =  np.sum((p0[z1:z2]-a*p[z1:z2])**2 )
            sys.stderr.write("# s = %g\n"%(s)) 
            return s

        res = minimize_scalar(myFunc,method='Brent')

        return res.x*p, res.x,res.fun

def main_all():

        # COMPUTATION PARAMETERS ----------------------------------------------
        path       = sys.argv[1]       # path to file storing direct data
        for fName in os.listdir(path):

                # FETCH SYNTHETIC SIGNAL pz FROM FILE ---------------------------------
                z,p0,pz,wD,D    = fetchSyntheticSignal(path+fName)

                # LOW-LEVEL PREDICTOR
                N = z.size
                dt = z[1] - z[0]
                pn1  = np.zeros(pz.size)
                pn1,cn1,nIt1 = sourceReconstruction(pn1,pz,wD,dt,N)

                # MINIMIZE Y-DISTANCE BETWEEN P AND P0 -------------------------------- 
                p0 = p0/sum(p0)
                zMin=0.02
                zMax=0.08
                pn1,sFac,ssq = yNormCurve(p0,pn1,(z,zMin,zMax))
                pz=sFac*pz

                print z[0],D, nIt1,ssq, "#", fName

def main():

        # COMPUTATION PARAMETERS ----------------------------------------------
        fName       = sys.argv[1]       # path to file storing direct data

        # FETCH SYNTHETIC SIGNAL pz FROM FILE ---------------------------------
        z,p0,pz,wD    = fetchSyntheticSignal(fName)

        # LOW-LEVEL PREDICTOR
        N = z.size
        dt = z[1] - z[0]
        pn1  = np.zeros(pz.size)
        pn1,cn1,nIt1 = sourceReconstruction(pn1,pz,wD,dt,N)

        # MINIMIZE Y-DISTANCE BETWEEN P AND P0 -------------------------------- 
        p0 = p0/sum(p0)
        zMin=0.02
        zMax=0.08
        pn1,sFac,ssq = yNormCurve(p0,pn1,(z,zMin,zMax))
        pz=sFac*pz

        print "# SIMULATION RESULTS --------------------------------------"
        for i in range(N):
           print z[i], p0[i], pz[i], pn1[i]

        print "# PL     : (zD) (pIni=0)"
        print "# PL-ITER:", z[0],nIt1
        print "# PL-MSE :", z[0],ssq


main_all()
