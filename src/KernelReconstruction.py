""" FILE: VolterraKernelReconstruction.pyx

Module implementing functions for solving the Volterra kernel reconstruction
problem, related to the simplified optoacoustic problem on the beam-axis 
assuming a Gaussian irradiation source profile, see Eq. (40) in section 1.3 
"Diffraction of laser-excited acoustic pulses" in the article

  Time-resolved laser optoacoustic tomography of inhomogeneous media,
  Karabutov, A.A and Podymova, N.B. and Letokhov, V.S., 
  Appl. Phys. B 63 (1996) 545-563

Here, the kernel reconstruction is achieved by fitting an effective 
parameterized approximated Volterra kernel to the exact Volterra Operator
of a set of reference curves.

Author: O. Melchert
Date:   28.03.2016
"""
import sys
from scipy.optimize import fmin,minimize_scalar,minimize,root
from VolterraKernelReconstruction import *

def kernelReconstruction((p0,pz),(a,K),(dt,Nt)):
        """optimization formulation of Volterra kernel reconstruction problem

        implements Powell's conjugate direction method for the minimization of
        the sum of squared residuals (SSR) measuring the deviation between 
        the exact and approximate Volterra operator for Volterra integral 
        equation of 2nd kind

        f(t) = \int_a^t K(t,s) f(s)~ds + g(t)

        NOTE:
        -# exact Volterra operator: (f(t)-g(t))
        -# approximate Volterra operator:  \int_a^t K(t,s) f(s)~ds 
           (based on approximation of the Volterra kernel K(t,s))

        \param[in]  p0    Reference curve: initial stress profile 
        \param[in]  pz    Reference curve: measured optoacoustic signal 
        \param[in]  a     Sequence of nonoptimal Fourier expansion coefficients 
        \param[in]  K     External function implementing approximate kernel
        \param[in]  dt    Time increment 
        \param[in]  Nt    Number of interpolation point along time axis 
        \param[out] aStar Optimzed expansion coefficients 
        """

        def myFunc(a):
            ssr = VolterraInt_SSR(pz,p0,a,K,dt,Nt)
            sys.stderr.write("# ssr = %g\n"%(ssr)) 
            return ssr

        res = minimize( myFunc,                     
                        a,                          
                        method = 'Powell',          
                        options = { 'xtol': 0.00001,  
                                    'ftol': 0.00001,  
                                    'maxiter': 5000 
                                  } 
                        )

        print "# KERNEL RECONSTRUCTION PROCEDURE"
        print "# succ  = ", res.success
        print "# stat  = ", res.status
        print "# msg   = ", res.message
        print "# Niter = ", res.nit
        print "KR-RES  (Na,SSQ) ",len(res.x), myFunc(res.x)

        return res.x

# EOF: KernelReconstruction.py
