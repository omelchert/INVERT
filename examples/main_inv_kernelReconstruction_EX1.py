""" FILE:  main_inv_kernelReconstruction_EX1.py

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
-# layered media - single layer
-# script generates data used in FIGS 1a and 1b

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


def meanSquareError(A,B):
        return ((A-B)**2).mean()


def main_kernelReconstruction_EX1():

        cmd    = sys.argv
        # SIMULATION PARAMETER ------------------------------------------------
        zMin   = 0.                            # medium initial point
        zMax   = 0.3                           # medium final point
        N      = 300                           # number of mesh-points 
        a0     = float(cmd[1])                 # isp width parameter
        zD     = float(cmd[2])                 # detector position
        # ABSORBING LAYER -----------------------------------------------------
        zAMin  = 0.1
        zAMax  = 0.2
        ma     = 24.
        # CHARACTERISTIC OPTOACOUSTIC PARAMETERS ------------------------------
        c0     = 1.                             # sonic velocity 
        wD     = 2*abs(zD)*c0/(a0*a0)   
        D      = wD/(c0*ma) 
        # KERNEL RECONSTRUCTION PARAMETERS ------------------------------------
        rTs    = float(sys.argv[3]) 
        Na     = int(sys.argv[4])
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
        print "# IRRADIATION SOURCE PROFILE ----------------------------------"
        print "# a0    =",a0
        print "# DETECTOR POSITION -------------------------------------------"
        print "# zD    =",zD
        print "# OPTOACOUSTIC PARAMETERS -------------------------------------"
        print "# c0    =",c0
        print "# wD    =",wD
        print "# D     =",D
        print "# RECONSTRUCTION PARAMETERS -----------------------------------"
        print "# rTs   =",rTs
        print "# Na    =",Na
        print "# ITERATIONS IN PICARD APPROXIMATION --------------------------"
        print "# M     =",M

        # SET BEAM AXIS -------------------------------------------------------
        (zAxis,dz) = np.linspace(zMin,zMax,N, retstep=True)

        # SET TIME RANGE FOR "NUMERICAL EXPERIMENT" --------------------------- 
        tAxis, dt = zAxis/c0, dz/c0 
       
        # INITIAL PRESSURE PROFILE - BEER LAMBERT (ISP-VIEW) ------------------
        p0 = iniStressProfile((zAxis,dz),(zAMin,zAMax),ma)

        # SOLVE DIRECT PROBLEM TO OBTAIN DETECTOR SIGNAL ----------------------
        pz  = OAVolterra_direct(p0,wD,dt,N)

        # KERNEL RECONSTRUCTION -----------------------------------------------
        ## set kernel
        K   = lambda a1,a2,a3,a4,a5: fourierKernel(a1,a2,a3,a4,a5,rTs)
        ## obtain initial sequence of expansion coefficients
        a0  = np.asarray([fourierCoeff(i,wD,dt,N,rTs) for i in range(Na)])
        ## fit parameters to reference curves
        a   = kernelReconstruction((p0,pz),(a0,K),(dt,N))

        print "# FOURIER COEFFICIENTS OF KERNEL EXPANSION"
        print "# Na = ", Na
        print "# (i) (a0[i]) (aOpt[i])"
        for i in range(Na):
                print "a ", i, a0[i], a[i]

        print "# (z) (Gauss approx) (K a0) (K aOpt)"
        for i in range(N):
                print "k ",zAxis[i],wD*np.exp(-wD*i*dt),K(a0,dt,i,0,N),K(a,dt,i,0,N)

        # PICARD-LINDELOEF ITERATION ------------------------------------------ 
        ## initial predictor
        fn = np.ones(N)
        ## correction procedure
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


main_kernelReconstruction_EX1()
# EOF: main_inv_kernelReconstruction_EX1.py
