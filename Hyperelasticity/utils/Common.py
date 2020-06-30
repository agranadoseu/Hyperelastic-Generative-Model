import numpy as np
from sympy import *


""" General """
kPa = 1000
sample = 100
lambdaRange = np.asarray([0.0001, 5.0])
# lambdaRange = np.asarray([0.0001, 2.0])
# lambdaRange = np.asarray([0.7, 1.3])
nP = 6  # postulates

# Anatomy type
GM = 0  # grey matter
WM = 1  # white matter
NH = 2  # normal healthy
AB = 3  # abnormal

# Model type
MRE = 0 # elastography
LIN = 1 # linear
HYP = 2 # hyperelastic


def sym_Stretches():
    # define stretches
    l1, l2, l3 = symbols('l1 l2 l3')

    return l1, l2, l3


def sym_Invariants(_l1, _l2, _l3):
    # compute invariants
    Ic = _l1**2 + _l2**2 + _l3**2
    IIc = _l1**4 + _l2**4 + _l3**4
    IIIc = _l1**2 * _l2**2 * _l3**2

    return Ic, IIc, IIIc



