""" FILE: OAVolterra_convolutionKernel.py

Module implementing functions for solving the Volterra equation of 2nd order
related to the simplified optoacoustic problem on the beam-axis assuming a
Gaussian irradiation source profile, see Eq. (40) in section 1.3 "Diffraction
of laser-excited acoustic pulses" in the article

  Time-resolved laser optoacoustic tomography of inhomogeneous media,
  Karabutov, A.A and Podymova, N.B. and Letokhov, V.S., 
  Appl. Phys. B 63 (1996) 545-563

The procedure for the numerical integration is implemented following Eqs.
18.2.1-4 in Chapter 18. "Integral Equations and Inverse Theory" of 

  Numerical Recipes in FORTRAN 77: The Art of Scientific Computing 
  Press, W.H. and Flannery, B.P. and Teukolsky, S.A. and Vetterling, W.T.,  
  Cambridge University Press, 1992

Further, since the optoacoustic (OA) pulse propagator is of convolution 
type, an efficient recursion formulation can be set up solving the 
forward problem with time complexity O(N).

Author: O. Melchert
Date:   08.03.2016
"""
import numpy as np

def iniStressProfile((z,dz),(zMin,zMax),ma):
        """initial acoustic stress profile

        \param[in]  z    z-axis
        \param[in]  dz   axial increment
        \param[in]  zMin start of new tissue layer 
        \param[in]  zMax end of new tissue layer 
        \param[in]  ma   absorption coefficient 
        \param[out] p0   initial stress profile 
        """
        mu_z = np.zeros(z.size)
        mu_z[int((zMin-z[0])/dz):int((zMax-z[0])/dz)] = ma
        return mu_z*np.exp(-np.cumsum(mu_z*dz))


def OAVolterra_direct(p0,wD,dt,Nt):
        """Forward (direct) solver for optoacoustic Volterra equation 

        Implements trapezoidal method for FORWARD solution of the 
        optoacoustic (OA)  Volterra equation

        pz(t) = p0(t) - \int_0^t K(t-s) p0(s)~ds;  K(t-s) = wD*exp(-wD*(t-s))

        where 

        p0(t)  = initial pressure profile at boundary of absorbing medium
        pz(t)  = optoacoustic signal at detection point
        K(t-s) = optoacoustic pulse propagator mediating diffraction 
                 transformation of optoacoustic signal

        NOTE #1: 
        Coincidentially, the integral kernel defining the OA pulse propagator
        is of convolution type, i.e. K(t,s)=K(t-s). Hence, so as to speed up 
        the computation, the term describing the diffraction transformation of 
        the OA signal can be represented by a recurrence relation. 
        The numerical value of the integral for successive values of t can then 
        efficientlty be calculated using memoization. The full calculation can 
        thus be carried out in time O(Nt).

        NOTE #2: 
        A general O(Nt^2) solution sheme for the 2nd kind Volterra Equation is 
        outlined in Eqs. 18.2.3 in Chapter 18. "Integral Equations and Inverse 
        Theory" of Numerical Recipes in Fortran 77.

        \param[in]  p0 initial pressure profile at absorber boundary 
        \param[in]  wD characteristic diffraction frequency 
        \param[in]  dt uniform mesh spacing 
        \param[in]  Nt number of meshpoints
        \param[out] pz optoacoustic signal at detection point  
        """
        # INITIALIZATION ------------------------------------------------------
        pz        = np.zeros(Nt)                # oa signal at detection point
        K0        = wD                          # oa propagator: K(0,0) 
        K1        = wD*np.exp(-wD*dt)           # oa propagator: K(1,0) 
        K1_K0     = np.exp(-wD*dt)              # quotient: K(i+1)/K(i)

        # SOLVE FORWARD PROBLEM VIA RECURRENCE RELATION -----------------------
        I     = 0               
        pz[0] = p0[0]           
        for i in range(1,Nt):
            I     = I*K1_K0 + 0.5*dt*(K1*p0[i-1] + K0*p0[i])
            pz[i] = p0[i] - I
        return pz


def OAVolterra_inverse(pz,wD,dt,Nt):
        """Inverse solver for optoacoustic Volterra equation 

        Implements trapezoidal method for INVERSE solution of the 
        optoacoustic (OA)  Volterra equation

        pz(t) = p0(t) - \int_0^t K(t-s) p0(s)~ds;  K(t-s) = wD*exp(-wD*(t-s))

        where 

        p0(t)  = initial pressure profile at boundary of absorbing medium
        pz(t)  = optoacoustic signal at detection point
        K(t-s) = optoacoustic pulse propagator mediating diffraction 
                 transformation of optoacoustic signal

        NOTE #1: 
        Coincidentially, the integral kernel defining the OA pulse propagator
        is of convolution type, i.e. K(t,s)=K(t-s). Hence, so as to speed up 
        the computation, the term describing the diffraction transformation of 
        the OA signal can be represented by a recurrence relation. 
        The numerical value of the integral for successive values of t can then 
        efficientlty be calculated using memoization. The full calculation can 
        thus be carried out in time O(Nt).

        NOTE #2: 
        A general O(Nt^2) solution sheme for the 2nd kind Volterra Equation is 
        outlined in Eqs. 18.2.3 in Chapter 18. "Integral Equations and Inverse 
        Theory" of Numerical Recipes in Fortran 77.

        \param[in]  pz optoacoustic signal at detection point  
        \param[in]  wD characteristic diffraction frequency 
        \param[in]  dt uniform mesh spacing 
        \param[in]  Nt number of meshpoints
        \param[out] p0 initial pressure profile at absorber boundary 
        """
        # INITIALIZATION ------------------------------------------------------
        p0        = np.zeros(Nt)                # oa signal at detection point
        K0        = wD                          # oa propagator: K(0,0) 
        K1        = wD*np.exp(-wD*dt)           # oa propagator: K(1,0) 
        K1_K0     = np.exp(-wD*dt)              # quotient: K(i+1)/K(i)

        # SOLVE INVERSE PROBLEM VIA RECURRENCE RELATION -----------------------
        I     = 0               
        p0[0] = pz[0]           
        for i in range(1,Nt):
            # USE INFO FROM RECONSTRUCTION STEP i-1 TO COMPUTE p0[i] ----------
            p0[i] = (pz[i] + (I + 0.5*dt*K0*p0[i-1])*K1_K0)/(1.-0.5*dt*K0)
            # ADVANCE DIFFRACTION TERM TO NEXT TIMESTEP -----------------------
            I     = I*K1_K0 + 0.5*dt*(K1*p0[i-1] + K0*p0[i])
        return p0


def OAVolterra_inverse_PicardIteration(pn,pz,wD,dt,Nt):
        """Inverse solver for optoacoustic Volterra equation based on the 
        method of successive approximations

        Implements successive approximation method for INVERSE solution of the 
        optoacoustic (OA)  Volterra equation

        p0_{n+1}(t) = pz(t) + \int_0^t K(t-s) p0_n(s)~ds
       
        K(t-s) = wD*exp(-wD*(t-s))

        where 

        p0_n(t) = n-th picard approximation of initial pressure profile 
        pz(t)   = optoacoustic signal at detection point
        K(t-s)  = optoacoustic pulse propagator mediating diffraction 
                 transformation of optoacoustic signal

        Since the integral kernel is convolution type, the term desribing the 
        diffraction transformation of the OA signal can be represented by a 
        recurrence relation and the numerical value of the integral for 
        successive values of t can then efficientlty be calculated using 
        memoization. The full calculation of advancing from the n-th Picard
        approximation to the n+1-th can thus be carried out in time O(Nt).
        As n -> \infty one expects p0_n -> p0. 

        The method of successive approximations is a correction procedure. So
        as to complete the numerical method, we also need to specify a 
        prediction procedure. Therefore distinguish:

        (1) FAR-FIELD RECONSTRUCTION:
        -# HIGH-PRECISION PREDICTOR: initial guess p0n obtained using the 
           Frauenhofer zone reconstruction procedure, i.e. the function
           OAVolterra_FrauenhoferZone_inverse()
        -# LOW-PRECISION PREDICTOR: p0n = const.

        (2) NEAR-FIELD RECONSTRUCTION:
        -# HIGH-PRECISION PREDICTOR: initial guess p0n = pz 
        -# LOW-PRECISION PREDICTOR: p0n = const.

        \param[in]  pn n-th Picard approximation to initial pressure profile 
        \param[in]  pz optoacoustic signal at detection point  
        \param[in]  wD characteristic diffraction frequency 
        \param[in]  dt uniform mesh spacing 
        \param[in]  Nt number of meshpoints
        \param[out] pn1 n+1-th Picard approximation to initial pressure profile 
        \param[out] cn Chebychev norm of pn and pn1 
        """
        # INITIALIZATION ------------------------------------------------------
        pn1       = np.zeros(Nt)                # n+1-th Picard approximation
        K0        = wD                          # oa propagator: K(0,0) 
        K1        = wD*np.exp(-wD*dt)           # oa propagator: K(1,0) 
        K1_K0     = np.exp(-wD*dt)              # quotient: K(i+1)/K(i)

        # SOLVE FORWARD PROBLEM VIA RECURRENCE RELATION -----------------------
        I     = 0               
        pn1[0] = pz[0]           
        for i in range(1,Nt):
            I      = I*K1_K0 + 0.5*dt*(K1*pn[i-1] + K0*pn[i])
            pn1[i] = pz[i] + I
        
        # COMPUTE CHEBYCHEV NORM OF n-TH AND n+1-TH PICARD APPROXIMATION ------
        cn = max(abs(pn1-pn))
        return pn1, cn


def OAVolterra_FrauenhoferZone_direct(p0,wD,dt,Nt):
        """Forward (direct) solver for optoacoustic Volterra equation 
        in Frauenhofer zone

        Implements centered difference approximation to initial pressure
        profile to yield optoacoustic (OA) signal in the Frauenhofer zone, i.e 
        for wD*ta >> 1 and z -> infty

        pFZ(t) = 1/wD dp0(t)/dt

        where 

        p0(t)  = initial pressure profile at boundary of absorbing medium
        pFZ(t)  = optoacoustic signal at detection point in Frauenhofer zone

        \param[in]  p0 initial pressure profile at absorber boundary 
        \param[in]  wD characteristic diffraction frequency 
        \param[in]  dt uniform mesh spacing 
        \param[in]  Nt number of meshpoints
        \param[out] pFZ optoacoustic signal in Frauenhofer zone
        """
        pFZ = np.zeros(Nt)
        for i in range(1,Nt-1):
            pFZ[i]=(p0[i+1]-p0[i-1])
        return pFZ/(2.*dt*wD)


def OAVolterra_FrauenhoferZone_inverse(pz,wD,dt,Nt):
        """Inverse solver for optoacoustic Volterra equation 
        in Frauenhofer zone

        Implements simple quadrature to reconstruct initial pressure
        profile from optoacoustic (OA) signal in the Frauenhofer zone, i.e 
        for wD*ta >> 1 and z -> infty, where

        pFZ(t) = 1/wD dp0(t)/dt

        where 

        p0(t)  = initial pressure profile at boundary of absorbing medium
        pFZ(t)  = optoacoustic signal at detection point in Frauenhofer zone

        \param[in]  p0 initial pressure profile at absorber boundary 
        \param[in]  wD characteristic diffraction frequency 
        \param[in]  dt uniform mesh spacing 
        \param[in]  Nt number of meshpoints
        \param[out] pFZ optoacoustic signal in Frauenhofer zone
        """
        p0     = np.zeros(Nt)
        cumSum = 0.
        for i in range(1,Nt-1):
            cumSum += 0.5*(pz[i-1]+pz[i+1])
            p0[i]   = cumSum 
        return p0*2.*dt*wD


# EOF: OAVolterra_convolutionKernel.py
