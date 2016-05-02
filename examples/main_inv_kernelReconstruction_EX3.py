""" FILE:  main_inv_kernelReconstruction_EX3.py

Script performing exemplary solution of the Volterra kernel reconstruction
problem for the simplified optoacoustic problem on the beam-axis assuming a
Gaussian irradiation source profile, see Eq. (40) in section 1.3 "Diffraction
of laser-excited acoustic pulses" in the article

  Time-resolved laser optoacoustic tomography of inhomogeneous media,
  Karabutov, A.A and Podymova, N.B. and Letokhov, V.S., 
  Appl. Phys. B 63 (1996) 545-563

Here, the kernel reconstruction is achieved by fitting an effective 
parameterized approximated Volterra kernel to the exact Volterra Operator
of a set of reference curves.

NOTE:
-# layered media - double layer
-# script generates data used in FIG 1c

Author: O. Melchert
Date:   28.03.2016
"""
import sys
sys.path.append('../src/')
sys.path.append('../src_cython_VolterraKernelReconstruction/')
import numpy as np
from OAVolterra_convolutionKernel import *
from VolterraKernelReconstruction import *
from KernelReconstruction import *


def fetchKernelExpansionCoeff(fName):
        """fetchKernelExpansionCoeff

        parse file for kernel expansion coefficients

        \param[in]  fName path to the file containing optimized results
        \param[out] a numpy array containing expansion coefficients
        \param[out] N number of expansion coefficients
        \param[out] rTs truncation radius used during kernel optimization
        """
        a = []
        
        with open(fName) as f:
           for line in f:
              c = line.split()
              if len(c)>1 and c[0]=='a':
                      a.append(float(c[3]))
              if len(c)>1 and c[1]=='rTs':
                      rTs = float(c[3])
              if len(c)>1 and c[1]=='a0':
                      a0 = float(c[3])
              if len(c)>1 and c[1]=='zD':
                      zD = float(c[3])
              if len(c)>1 and c[1]=='c0':
                      c0 = float(c[3])

        return (np.asarray(a), len(a),rTs),(a0,zD,c0)


def p0Exp(p,r,rMin=0.3,rMax=0.6,ma=24.,A=0.):
        for i in range(len(r)):
                if r[i] > rMin and r[i] < rMax:
                        p[i]=ma*np.exp(- (r[i]-rMin)*ma - A  )
        return p

def meanSquareError(A,B):
        return ((A-B)**2).mean()

def main_kernelReconstruction_EX3():

        cmd    = sys.argv

        # KERNEL RECONSTRUCTION PARAMETERS ------------------------------------
        fName  = sys.argv[1]                    # file with expansion coeff         
        (a,Na,rTs),(a0,zD,c0) = fetchKernelExpansionCoeff(fName)

        # SIMULATION PARAMETER ------------------------------------------------
        zMin   = 0.                            # medium initial point
        zMax   = 0.3                           # medium final point
        N      = 300                           # number of mesh-points 
        # ABSORBING LAYER -----------------------------------------------------
        z1,z2,z3    = 0.10,0.15,0.22
        z01,z02,ma0 = z1, z2, 24.
        z11,z12,ma1 = z2, z3, 12.
        zAMin  = z01
        zAMax  = z12
        # CHARACTERISTIC OPTOACOUSTIC PARAMETERS ------------------------------
        c0     = 1.                             # sonic velocity 
        wD     = 2*abs(zD)*c0/(a0*a0)   
        D      = wD/(c0*ma0)
        # ITERATIONS IN PICARD APPROXIMATION ----------------------------------
        M      = 100


        # PRINT SIMULATION DETAILS --------------------------------------------
        print "# FILE: ", cmd[0] 
        print "# SIMULATION PARAMETERS ---------------------------------------"
        print "# zMin  =",zMin
        print "# zMax  =",zMax
        print "# p0-PROFILE --------------------------------------------------"
        print "# zAMin =",zAMin
        print "# zAMax =",zAMax
        print "# N     =",N
        print "# rECONSTRUCTION PARAMETERS -----------------------------------"
        print "# a0    =",a0
        print "# zD    =",zD
        print "# c0    =",c0
        print "# rTs   =",rTs
        print "# Na    =",Na
        print "# OPTOACOUSTIC PARAMETERS -------------------------------------"
        print "# wD    =",wD
        print "# D     =",D
        print "# ITERATIONS IN PICARD APPROXIMATION --------------------------"
        print "# M     =",M

        # SET BEAM AXIS -------------------------------------------------------
        (zAxis,dz) = np.linspace(zMin,zMax,N, retstep=True)

        # SET TIME RANGE FOR "NUMERICAL EXPERIMENT" --------------------------- 
        tAxis, dt = zAxis/c0, dz/c0 
       
        # INITIAL PRESSURE PROFILE - BEER LAMBERT -----------------------------
        p0 = np.zeros(N)
        p0 = p0Exp(p0,zAxis, z01, z02, ma0, 0.)
        p0 = p0Exp(p0,zAxis, z11, z12, ma1, (z02-z01)*ma0)

        # SOLVE DIRECT PROBLEM TO OBTAIN DETECTOR SIGNAL ----------------------
        pz = OAVolterra_direct(p0,wD,dt,N)

        K   = lambda a1,a2,a3,a4,a5: fourierKernel(a1,a2,a3,a4,a5,rTs)

        print "# FOURIER COEFFICIENTS OF KERNEL EXPANSION"
        print "# Na = ", Na
        print "# (i) (aOpt[i])"
        for i in range(Na):
                print "a ", i, a[i]

        print "# (z) (Gauss approx) (K aOpt)"
        for i in range(N):
                print "k ",zAxis[i],wD*np.exp(-wD*i*dt),K(a,dt,i,0,N)

        fn = np.ones(N)
        for m in range(1,1+M):
                I = VolterraOperator(fn,a,K,dt,N)
                fn = pz + I
                print "# SUCCESSIVE APPROXIMATION OF ORDER", m
                print "SA-MSE (m,MSE) ", m,meanSquareError(p0,fn)
                print "# (t) (c tau) (p0) (pz) (p0_rec_succApprox)"
                for i in range(N):
                  print "m%d"%(m)\
                      ,tAxis[i]\
                      ,zAxis[i]-zAMin\
                      ,p0[i]\
                      ,pz[i]\
                      ,fn[i]
                print


main_kernelReconstruction_EX3()
# EOF:  main_inv_kernelReconstruction_EX3.py
