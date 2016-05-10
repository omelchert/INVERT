""" FILE:  main_inv_EX2.py

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
import numpy as np
from OAVolterra_convolutionKernel import *

def iniStressProfile((z,dz),(zMin,zMax)):
        """initial acoustic stress profile

        \param[in]  z    z-axis
        \param[in]  dz   axial increment
        \param[in]  zMin start of new tissue layer 
        \param[in]  zMax end of new tissue layer 
        \param[out] p0   initial stress profile 
        """
        p0 = np.zeros(z.size)
        p0[int((zMin-z[0])/dz):int((zMax-z[0])/dz)] = 1.0
        return p0 


def correctFwdMode(zD,p0):
        if zD > 0:
           zD = -zD
           p0 = p0[::-1]
        return zD,p0

def sourceReconstruction(pn,pz,wD,dt,N):
        cn  = 1.0
        nIt = 0
        while(cn>0.000001):
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


def MSE(p0,pR):
        return ((p0-pR)**2).mean()


def OAVolterra_inversionViaResolventKernel(pD,wD,dt,N):
        pRec = np.zeros(pD.size)
        for i in range(pD.size):
           pRec[i] = pD[i]+wD*np.trapz(pD[:i],dx=dt)
        return pRec

def main_resolvent():

        cmd    = sys.argv
        # SIMULATION PARAMETER ------------------------------------------------
        zMin   = 0.0                            # medium initial point
        zMax   = 0.3                           # medium final point
        N      = 500                          # number of mesh-points 
        a0     = 0.1
        zD     = float(cmd[1])                 # detector position
        # ABSORBING LAYER -----------------------------------------------------
        zAMin  = 0.1
        zAMax  = 0.2
        # CHARACTERISTIC OPTOACOUSTIC PARAMETERS ------------------------------
        c0     = 1.                             # sonic velocity 
        wD     = 2*abs(zD)*c0/(a0*a0)   
        D      = abs(zD)*(zAMax-zAMin)/(a0*a0*np.pi) 

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
        print "# MATERIAL PARAMETERS -----------------------------------------"
        print "# c0    =",c0
        print "# OPTOACOUSTIC PARAMETERS -------------------------------------"
        print "# wD    =",wD
        print "# D     =",D

        # SET BEAM AXIS -------------------------------------------------------
        (zAxis,dz) = np.linspace(zMin,zMax,N, retstep=True)

        # SET TIME RANGE FOR "NUMERICAL EXPERIMENT" --------------------------- 
        tAxis, dt = zAxis/c0, dz/c0 
       
        # INITIAL PRESSURE PROFILE - BEER LAMBERT (ISP-VIEW) ------------------
        p0 = iniStressProfile((zAxis,dz),(zAMin,zAMax))

        # SOLVE DIRECT PROBLEM TO OBTAIN DETECTOR SIGNAL ----------------------
        pD = OAVolterra_direct(p0,wD,dt,N)

        # PICARD-LINDELOEF ITERATION ------------------------------------------
        # LOW-LEVEL PREDICTOR
        pn1  = np.zeros(pD.size)
        pn1,cn1,nIt1 = sourceReconstruction(pn1,pD,wD,dt,N)
        MSE1 = MSE(p0,pn1)
        # HIGH-LEVEL NEAR-FIELD PREDICTOR
        pn2  = pD
        pn2,cn2,nIt2 = sourceReconstruction(pn2,pD,wD,dt,N)
        MSE2 = MSE(p0,pn2)
        # HIGH-LEVEL FAR-FIELD PREDICTOR
        pn3  = quadrature(pD,tAxis)*wD  
        pn3,cn3,nIt3 = sourceReconstruction(pn3,pD,wD,dt,N)
        MSE3 = MSE(p0,pn3)

        print "# SIMULATION RESULTS --------------------------------------"
        for i in range(1,N-1):
                print zAxis[i]-zAMin, p0[i], pD[i],  pn1[i], pn2[i], pn3[i]
        
        print "# PL     : (pIni=0) (pIni=pD) (pIni=Int(pD))"
        print "# PL-ITER:", nIt1,nIt2,nIt3
        print "# PL-MSE :", MSE1,MSE2,MSE3

main_resolvent()
