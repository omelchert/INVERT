1. INTRODUCTION

INVERT - InversioN Via voltErra keRnel reconsTruction

implements python modules for the direct and inverse simulation of optoacoustic
signals in the paraxial approximation to the optoacoustic wave equation.
Therein, the calculation of the excess pressure is accomplished via numerical
solution of the optoacoustic Volterra integral equation. For the inverse
solution, two distinct inverse problems are considered:

I.1: SOURCE RECONSTRUCTION - reconstruct initial stress profile from measured
optoacoustic signal upon knowledge of the mathematical model that mediates the
underlying diffraction transformation

I.2: KERNEL RECONSTRUCTION - reconstruct stress wave propagator to account for
the apparent diffraction transformation shown by the optoacoustic signal 

Use-cases that illustrate the inversion of optoacoustic signals for layered
media on the beam axis are detailed in the examples folder. 

So as to avoid unnecessary overhead, the implementation follows a procedural
programming style.


2. DEPENDENCIES

INVERT requires the functionality of NumPy, a fundamental package for
scientific computing (see www.numpy.org). Time-critical parts of INVERT are
implemented in cython, an optimising static compiler for python (see
www.cython.org).


3. CONTENT

README.txt      
LICENSE.txt     

src/
  KernelReconstruction.py
  OAVolterra_convolutionKernel.py

src_cython_VolterraKernelReconstruction/
  Makefile
  VolterraKernelReconstruction.pyx
  setup.py

examples/
  FIG1ab/
     getSSQ.sh
     kRec_*.dat
     SSQ_zD-0.02_rTs0.20.dat
     SSQ_zD-0.5_rTs0.10.dat
     ssq_R.dat
     GP/
        FIG1a_kRec_zD-0.5.gpi
        FIG1b_kRec_zD-0.5.gpi
        FIGS/
           FIG1a_kRec_zD-0.5.eps
           FIG1a_kRec_zD-0.5.pdf
           FIG1b_kRec_zD-0.5_Na51.eps
           FIG1b_kRec_zD-0.5_Na51.pdf

  FIG1c/
     doubleLayer_*.dat
     GP/
        FIG1c_doubleLayer_kRec.gpi
        FIGS/
           FIG1c_sourceRec_doubleLayer.eps
           FIG1c_sourceRec_doubleLayer.pdf

  FIG2a/
     INPUT/
        topHat_*.dat
     kRec_*.dat
     GP/
        FIG2a_kRec_topHat_zD-0.5.gpi
        FIGS/
           FIG2a_kRec_topHat_zD-0.5_a0.10_R0.10.eps
           FIG2a_kRec_topHat_zD-0.5_a0.10_R0.10.pdf

  getData_FIG1ab.sh
  getData_FIG1c.sh
  getData_FIG2a.sh
  getData_FIG2b.sh
  main_inv_kernelReconstruction_EX1.py
  main_inv_kernelReconstruction_EX2.py
  main_inv_kernelReconstruction_EX3.py


4. LICENSE

BSD 3-Clause License


5. ACKNOWLEDGEMENTS

O. Melchert acknowledges support from the VolkswagenStiftung within the
Nieders\"achsisches Vorab program in the framework of the project Hybrid
Numerical Optics. 


