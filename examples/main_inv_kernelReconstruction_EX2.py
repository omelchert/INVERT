""" FILE:  main_inv_kernelReconstruction_EX2.py

Script performing exemplary solution of the Volterra kernel reconstruction
problem considering a top-hat irradiation source profile

Here, the kernel reconstruction is achieved by fitting an effective 
parameterized approximated Volterra kernel to the exact Volterra Operator
of a set of reference curves.

NOTE:
-# layered media - single layer
-# irradiation source profile: top-hat
-# script generates data used in FIG 2a

Author: O. Melchert
Date:   28.03.2016
"""
import sys
sys.path.append('../src/')
sys.path.append('../src_cython_VolterraKernelReconstruction/')
import numpy as np
from VolterraKernelReconstruction import *
from KernelReconstruction import *

def fetchInput(fName):
        z  = []
        p0 = []
        pz = []
        with open(fName,'r') as f:
           for line in f:
               c = line.split()
               if len(c)>2 and line[0]!='#':
                   z.append(float(c[1]))                   
                   p0.append(float(c[2]))                   
                   pz.append(float(c[3]))                   
        return np.asarray(z),np.asarray(p0),np.asarray(pz)

def meanSquareError(A,B):
        return ((A-B)**2).mean()

def main_kernelReconstruction_EX2():

        fName  = sys.argv[1]
        # KERNEL RECONSTRUCTION PARAMETERS ------------------------------------
        rTs    = float(sys.argv[2]) 
        Na     = int(sys.argv[3])
        # ITERATIONS IN PICARD APPROXIMATION ----------------------------------
        M      = 100

        
        zAxis, p0, pz = fetchInput(fName)        
        N  = zAxis.size
        dt = zAxis[1]-zAxis[0]

        wD = 2*0.5/0.1/0.1 

        K   = lambda a1,a2,a3,a4,a5: fourierKernel(a1,a2,a3,a4,a5,rTs)
        a0  = np.asarray([fourierCoeff(i,wD,dt,N,rTs) for i in range(Na)])
        a   = kernelReconstruction((p0,pz),(a0,K),(dt,N))

        print "# FOURIER COEFFICIENTS OF KERNEL EXPANSION"
        print "# Na = ", Na
        print "# (i) (a0[i]) (aOpt[i])"
        for i in range(Na):
                print "a ", i, a0[i], a[i]

        print "# (z) (Gauss approx) (K a0) (K aOpt)"
        for i in range(N):
                print "k ",zAxis[i]-zAxis[0],wD*np.exp(-wD*i*dt),K(a0,dt,i,0,N),K(a,dt,i,0,N)

        fn = np.ones(N)
        for m in range(1,1+M):
                I = VolterraOperator(fn,a,K,dt,N)
                fn = pz + I
                print "# SUCCESSIVE APPROXIMATION OF ORDER", m
                print "SA-MSE (m,MSE) ", m,meanSquareError(p0,fn)
                print "# (c tau) (p0) (pz) (p0_rec_succApprox)"
                for i in range(N):
                  print "m%d"%(m)\
                      ,zAxis[i]\
                      ,p0[i]\
                      ,pz[i]\
                      ,fn[i]
                print

main_kernelReconstruction_EX2()
# EOF: main_inv_kernelReconstruction_EX2.py
