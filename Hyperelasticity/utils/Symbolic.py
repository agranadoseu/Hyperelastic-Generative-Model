import numpy as np
from sympy import *


def ComputeEnergyGradient(W, l1, l2, l3):
    # energy gradient
    dWdl1 = diff(W, l1)
    dWdl2 = diff(W, l2)
    dWdl3 = diff(W, l3)
    dW = np.array([dWdl1, dWdl2, dWdl3])
    # print(dWdl1)
    # print(dWdl2)
    # print(dWdl3)
    # print('Energy gradient dW = ')
    # print(dW)

    return dW


def ComputeEnergyHessian(dW, l1, l2, l3):
    # energy Hessian
    d2Wdl1dl1 = diff(dW[0], l1)
    d2Wdl1dl2 = diff(dW[0], l2)
    d2Wdl1dl3 = diff(dW[0], l3)
    d2Wdl2dl1 = diff(dW[1], l1)
    d2Wdl2dl2 = diff(dW[1], l2)
    d2Wdl2dl3 = diff(dW[1], l3)
    d2Wdl3dl1 = diff(dW[2], l1)
    d2Wdl3dl2 = diff(dW[2], l2)
    d2Wdl3dl3 = diff(dW[2], l3)
    H = np.array([[d2Wdl1dl1, d2Wdl1dl2, d2Wdl1dl3],
                  [d2Wdl2dl1, d2Wdl2dl2, d2Wdl2dl3],
                  [d2Wdl3dl1, d2Wdl3dl2, d2Wdl3dl3]])

    # print(d2Wdl1dl1)
    # print('Energy Hessian H = ')
    # print(H)

    return H



def ComputeCauchyStress(dW, l1, l2, l3):
    # Cauchy stress (sigma)
    sigmaW1 = l1 / (l1 * l2 * l3) * dW[0]
    sigmaW2 = l2 / (l1 * l2 * l3) * dW[1]
    sigmaW3 = l3 / (l1 * l2 * l3) * dW[2]
    sigma = np.array([sigmaW1, sigmaW2, sigmaW3])
    # print(sigmaW1)
    # print('Cauchy stress sigma = ')
    # print(sigma)

    return sigma