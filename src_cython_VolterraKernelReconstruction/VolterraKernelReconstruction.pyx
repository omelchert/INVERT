""" FILE: VolterraKernelReconstruction.pyx

Module implementing functions for solving the Volterra kernel reconstruction
problem, related to the simplified optoacoustic problem on the beam-axis 
assuming a Gaussian irradiation source profile, see Eq. (40) in section 1.3 
"Diffraction of laser-excited acoustic pulses" in the article

  Time-resolved laser optoacoustic tomography of inhomogeneous media,
  Karabutov, A.A and Podymova, N.B. and Letokhov, V.S., 
  Appl. Phys. B 63 (1996) 545-563

Author: O. Melchert
Date:   27.03.2016
"""
import  numpy as np
cimport numpy as np
cimport cython

# abbreviate new datatype for 64bit floats
ctypedef np.float64_t dtype_t

# prefer C function with small overhead 
cdef extern from "math.h":
        float exp(float value) 
        float cos(float value) 
        float sin(float value) 


def fourierCoeff(Py_ssize_t k, dtype_t wD, dtype_t dt, Py_ssize_t Nt, dtype_t xMax):
        """high-level estimator for Gaussian-beam Fourier coefficients

        \param[in]  k     integer order of Fourier coefficient
        \param[in]  wD    characteristic optoacoustic frequency 
        \param[in]  dt    time increment 
        \param[in]  Nt    number of interpolation point along time axis 
        \param[in]  xMax  finite cut-off distance
        \param[out] coeff Fourier expansion coefficient of order k 
        """
        cdef Py_ssize_t i,Nmax

        Nmax = int(xMax/dt)
        scl  = 2.0*3.14157/Nmax

        if k==0:
             n   = 0
             myF = np.cos
        elif k%2==0:
             n   = k/2
             myF = np.sin
        else:
             n   = (k+1)/2
             myF = np.cos

        mySum = 0.
        for i in range(Nmax):
            mySum += wD*exp(-wD*i*dt)*myF(n*i*scl)*dt

        return mySum*2./(Nmax*dt)


@cython.boundscheck(False)
@cython.wraparound(False)
def fourierKernel(np.ndarray[dtype_t, ndim=1] c, dtype_t dt,Py_ssize_t i,Py_ssize_t j, Py_ssize_t Nt, dtype_t xMax):
        """effective Volterra kernel

        \param[in]  c     sequence of Fourier expansion coefficients 
        \param[in]  dt    time increment 
        \param[in]  i     integer id time-coordinate (first argument)
        \param[in]  j     integer id time-coordinate (second argument)
        \param[in]  Nt    number of interpolation point along time axis 
        \param[in]  xMax  finite cut-off distance
        \param[out] Kval  num. val. of effective Volterra kernel at given point 
        """
        cdef dtype_t x, ms, scl

        scl  = 2.0*3.1415927/xMax
        x    = (i-j)*dt
        ms   = 0.0

        if x<xMax:
            ms   = c[0]/2
            for k in range(1,len(c)/2 + 1):
                ms += c[2*k-1]*cos(k*x*scl) + c[2*k]*sin(k*x*scl)

        return ms 


# Warning: the following features were disabled to yield speed-up
# (i) bound-checking for array indices
# (ii) support for negative array indices 
@cython.boundscheck(False)
@cython.wraparound(False)
def VolterraInt_SSR(np.ndarray[dtype_t, ndim=1] g, np.ndarray[dtype_t, ndim=1] f, np.ndarray[dtype_t, ndim=1] c, Kernel, dtype_t dt, Py_ssize_t Nt):
        """sum of squared residuals

        Implements trapezoidal method for the numerical calculation of the
        sum of squared residuals (SSR) measuring the deviation between 
        the exact and approximate Volterra operator for Volterra integral 
        equation of 2nd kind

        f(t) = \int_a^t K(t,s) f(s)~ds + g(t)

        outlined in Eqs. 18.2.2 - 18.2.4 in Chapter 18. "Integral 
        Equations and Inverse Theory" of Numerical Recipes in Fortran 77

        NOTE:
        -# exact Volterra operator: (f(t)-g(t))
        -# approximate Volterra operator:  \int_a^t K(t,s) f(s)~ds 
           (based on Fourier approximation of the Volterra kernel K(t,s))

        \param[in]  g   Input array containing rhs of the Volterra eqn
        \param[in]  f   Input array containing lhs of the Volterra eqn
        \param[in]  c   Sequence of Fourier expansion coefficients 
        \param[in]  K   External function implementing approximate kernel
        \param[in]  dt  Uniform mesh spacing 
        \param[in]  Nt  Number of meshpoints
        \param[out] SSR sum of squared residuals 
        """
        # DECLARATION ---------------------------------------------------------
        cdef Py_ssize_t i, j
        cdef dtype_t s
        cdef np.ndarray[dtype_t, ndim=1] Kt

        # INITIALIZATION ------------------------------------------------------
        Kt = np.zeros(Nt)   
        
        # LOOKUP TABLE FOR KERNEL ---------------------------------------------
        for i in range(1,Nt):
                Kt[i] = Kernel(c,dt,i,0,Nt)

        # PERFORM INTEGRATION -------------------------------------------------
        s = 0.
        for i in range(1,Nt):
            I_prt = 0.5*(Kt[i]*f[0]+Kt[0]*f[i])
            for j in range(1,i):
                I_prt += Kt[i-j]*f[j]
            I_prt *= dt
            
            # UPDATE SUM OF SQUARES ------------------------------------------- 
            s += (f[i] - g[i] - I_prt)**2

        return s


# Warning: the following features were disabled to yield speed-up
# (i) bound-checking for array indices
# (ii) support for negative array indices 
@cython.boundscheck(False)
@cython.wraparound(False)
def VolterraOperator(np.ndarray[dtype_t, ndim=1] f, np.ndarray[dtype_t, ndim=1] polCoeff, Kernel, dtype_t dt, Py_ssize_t Nt):
        """approximate Volterra operator

        Implements trapezoidal method for the numerical calculation of the
        approximate Volterra operator 

        (Kf)(t) = \int_a^t K(t,s) f(s)~ds 

        The Volterra operator (aka: diffraction oparator/term) conveys the
        effect of diffraction on the initial stress profile as a function
        of time.

        \param[in]  f   Input array containing lhs of the Volterra eqn
        \param[in]  polCoeff   Sequence of Fourier expansion coefficients 
        \param[in]  K   External function implementing approximate kernel
        \param[in]  dt  Uniform mesh spacing 
        \param[in]  Nt  Number of meshpoints
        \param[out] Kf  approximate Volterra operator 
        """
        # DECLARATION ---------------------------------------------------------
        cdef Py_ssize_t i, j
        cdef dtype_t s
        cdef np.ndarray[dtype_t, ndim=1] Kt
        cdef np.ndarray[dtype_t, ndim=1] I

        # INITIALIZATION ------------------------------------------------------
        Kt = np.zeros(Nt)   
        I  = np.zeros(Nt)   
        
        # LOOKUP TABLE FOR KERNEL ---------------------------------------------
        for i in range(1,Nt):
                Kt[i] = Kernel(polCoeff,dt,i,0,Nt)

        # PERFORM INTEGRATION -------------------------------------------------
        I[0] = 0
        for i in range(1,Nt):
            I_prt = 0.5*(Kt[i]*f[0]+Kt[0]*f[i])
            for j in range(1,i):
                I_prt += Kt[i-j]*f[j]
            I_prt *= dt
            I[i] = I_prt
            
        return I

# EOF: VolterraKernelReconstruction.pyx
