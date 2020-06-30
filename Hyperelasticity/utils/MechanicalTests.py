import numpy as np
from sympy import *

def Uniaxial():
    l = symbols('l')
    l_l1 = l
    l_l2 = l ** (-0.5)
    l_l3 = l ** (-0.5)
    # print(l)
    # print(l_l1)
    # print(l_l2)
    # print(l_l3)

    return l, l_l1, l_l2, l_l3