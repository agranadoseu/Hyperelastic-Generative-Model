"""
This function tests postulates of a strain energy density function
These are discussed in [Dirijani2010] and presented by Ogden and Treolar
"""

import numpy as np
import matplotlib.pyplot as plt

from Hyperelasticity.utils import Evaluate

def monotonic_strictly_increasing(V):
    return all(x<y for x, y in zip(V, V[1:]))

def monotonic_increasing(V):
    return all(x<=y for x, y in zip(V, V[1:]))

def monotonic_decreasing(V):
    return all(x>=y for x, y in zip(V, V[1:]))

def monotonic_strictly_decreasing(V):
    return all(x>y for x, y in zip(V, V[1:]))



def verify(l, _lambdas, _W, _dW, _H, _sigma):
    ''' postulates '''
    P1 = False
    P2 = False
    P3 = False
    P4 = False
    P5 = False
    P6 = False

    ''' Resting state '''
    print('Postulates @ Resting state:')
    # W is zero
    v = Evaluate.SinglePoint(_W, l, 1.0)
    print(v)
    if round(v,4) == 0.0:
        P1 = True
    print('     W(1)={:.20f} is zero [{}]'.format(v, P1))

    # dW is zero
    v0 = Evaluate.SinglePoint(_dW[0], l, 1.0)
    v1 = Evaluate.SinglePoint(_dW[1], l, 1.0)
    v2 = Evaluate.SinglePoint(_dW[2], l, 1.0)
    dWoffset = -v0
    print(v0)
    print(v1)
    print(v2)
    if round(v0, 4)==0.0 and round(v1, 4)==0.0 and round(v2, 4)==0.0:
        P2 = True
    print('     dW(1)=[{:.4f},{:.4f},{:.4f}] is zero [{}]'.format(v0, v1, v2, P2))

    # diag(H) > 0
    v0 = Evaluate.SinglePoint(_H[0,0], l, 1.0)
    v1 = Evaluate.SinglePoint(_H[1,1], l, 1.0)
    v2 = Evaluate.SinglePoint(_H[2,2], l, 1.0)
    if v0 > 0.0 and v1 > 0.0 and v2 > 0.0:
        P3 = True
    print('     diag(H)=[{:.4f},{:.4f},{:.4f}] is > 0 [{}]'.format(v0, v1, v2, P3))

    # det(H) > 0
    H1 = np.zeros((3,3))
    for i in range(3):
        for j in range(3):
            H1[i,j] = Evaluate.SinglePoint(_H[i, j], l, 1.0)
    v = np.linalg.det(H1)
    if v > 0.0:
        P4 = True
    print('     det(H)={:.4f} is > 0 [{}]'.format(v, P4))

    ''' Deforming state '''
    print('Postulates @ Deforming state:')
    # W is non-negative
    W_v, dW_v, H_v, sigma_v = Evaluate.Functions(l, _lambdas, _W, _dW, _H, _sigma)
    P5 = not any(v<0 for v in W_v)
    print('     W is non-negative [{}]'.format(P5))

    # dW is monotonicly increasing
    P6 = monotonic_increasing(dW_v[0])
    print('     dW is monotonic increasing [{}]'.format(P6))
    # print(_dW[0])
    # axis = plt.subplot(121)
    # plt.plot(_lambdas, W_v)
    # axis.set_xlim([0, 5])
    # axis.set_ylim([0, 4e7])
    # axis = plt.subplot(122)
    # plt.plot(_lambdas, dW_v[0]-dWoffset)
    # axis.set_xlim([-5, 5])
    # axis.set_ylim([-2e7, 2e7])
    # plt.show()
    # print(_lambdas)
    # print(W_v)
    # print(dW_v[0])
    # input('break')

    P = np.asarray([P1, P2, P3, P4, P5, P6])
    return P, dWoffset