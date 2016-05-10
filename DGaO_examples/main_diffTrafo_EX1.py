""" FILE:  main_diffTrafo_EX1.py

Script performing exemplary solution of the Volterra kernel reconstruction
problem for the simplified optoacoustic problem on the beam-axis assuming a
Gaussian irradiation source profile, see Eq. (40) in section 1.3 "Diffraction
of laser-excited acoustic pulses" in the article

  Time-resolved laser optoacoustic tomography of inhomogeneous media,
  Karabutov, A.A and Podymova, N.B. and Letokhov, V.S., 
  Appl. Phys. B 63 (1996) 545-563


NOTE:
-# box signal
-# script generates data used in FIG a 

Author: O. Melchert
Date:   09.05.2016
"""
import sys
sys.path.append('../src/')
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

def main_diffTrafo_EX1():

        cmd    = sys.argv
        # SIMULATION PARAMETER ------------------------------------------------
        zMin   = 0.                            # medium initial point
        zMax   = 0.3                           # medium final point
        N      = 300                           # number of mesh-points 
        a0     = float(cmd[1])                 # isp width parameter
        # ABSORBING LAYER -----------------------------------------------------
        zAMin  = 0.1
        zAMax  = 0.2
        # CHARACTERISTIC OPTOACOUSTIC PARAMETERS ------------------------------
        c0     = 1.                             # sonic velocity 

        # PRINT SIMULATION DETAILS --------------------------------------------
        print "# FILE: ", cmd[0] 
        print "# SIMULATION PARAMETERS ---------------------------------------"
        print "# zMin  =",zMin
        print "# zMax  =",zMax
        print "# p0-PROFILE --------------------------------------------------"
        print "# zAMin =",zAMin
        print "# zAMax =",zAMax
        print "# N     =",N
        print "# c0    =",c0
        print "# IRRADIATION SOURCE PROFILE ----------------------------------"
        print "# a0    =",a0

        # SET BEAM AXIS -------------------------------------------------------
        (zAxis,dz) = np.linspace(zMin,zMax,N, retstep=True)

        # SET TIME RANGE FOR "NUMERICAL EXPERIMENT" --------------------------- 
        tAxis, dt = zAxis/c0, dz/c0 
       
        # INITIAL PRESSURE PROFILE - BEER LAMBERT (ISP-VIEW) ------------------
        p0 = iniStressProfile((zAxis,dz),(zAMin,zAMax))

        for zD in np.linspace(-0.01,-1.0,50):
                # COMPUTE DIFFRACTION CHARACTERISTICS ---------------------------------
                wD     = 2*abs(zD)*c0/(a0*a0)   
                D      = abs(zD)*(zAMax-zAMin)/(a0*a0*np.pi) 
                print "# DETECTOR POSITION -------------------------------------------"
                print "# zD    =",zD
                print "# OPTOACOUSTIC PARAMETERS -------------------------------------"
                print "# wD    =",wD
                print "# D     =",D

                # SOLVE DIRECT PROBLEM TO OBTAIN DETECTOR SIGNAL ----------------------
                pz  = OAVolterra_direct(p0,wD,dt,N)

                print "# (t) (c tau) (pz)"
                for i in range(N):
                  print zD\
                      ,D\
                      ,zAxis[i]-zAMin\
                      ,pz[i]
                print


main_diffTrafo_EX1()
# EOF: main_diffTrafo_EX1.py
