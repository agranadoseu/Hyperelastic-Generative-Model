import numpy as np
from sympy import *

from Hyperelasticity.utils import Common

s = Common.sample

def Functions(l, stretches, _W, _dW, _H, _sigma):
    # plot(_W, (l, 0, 5), ylim=[0, 4e7])

    # convert functions
    f_W = lambdify(l, _W, "numpy")

    f_dW1 = lambdify(l, _dW[0], "numpy")
    f_dW2 = lambdify(l, _dW[1], "numpy")
    f_dW3 = lambdify(l, _dW[2], "numpy")

    f_H11 = lambdify(l, _H[0, 0], "numpy")
    f_H12 = lambdify(l, _H[0, 1], "numpy")
    f_H13 = lambdify(l, _H[0, 2], "numpy")
    f_H21 = lambdify(l, _H[1, 0], "numpy")
    f_H22 = lambdify(l, _H[1, 1], "numpy")
    f_H23 = lambdify(l, _H[1, 2], "numpy")
    f_H31 = lambdify(l, _H[2, 0], "numpy")
    f_H32 = lambdify(l, _H[2, 1], "numpy")
    f_H33 = lambdify(l, _H[2, 2], "numpy")

    f_sigma1 = lambdify(l, _sigma[0], "numpy")
    f_sigma2 = lambdify(l, _sigma[1], "numpy")
    f_sigma3 = lambdify(l, _sigma[2], "numpy")

    # evaluate
    v_W = f_W(stretches)
    v_dW = np.array([f_dW1(stretches), f_dW2(stretches), f_dW3(stretches)])
    v_H = np.array([[f_H11(stretches), f_H12(stretches), f_H13(stretches)],
                    [f_H21(stretches), f_H22(stretches), f_H23(stretches)],
                    [f_H31(stretches), f_H32(stretches), f_H33(stretches)]])
    v_sigma = np.array([f_sigma1(stretches), f_sigma2(stretches), f_sigma3(stretches)])

    # H
    full_H = np.zeros((3,3,s))
    for i in range(3):
        for j in range(3):
            # print('v_H[i, j]')
            # print(v_H[i, j])
            if v_H[i, j] is 0:
                # print('v_H is zero')                                    # zero value
                full_H[i, j] = np.full((s), v_H[i, i])
            else:
                n_ij = v_H[i, j].shape
                if not n_ij:
                    # print('v_H is a scalar = {:.2f}'.format(v_H[i, i]))       # one value
                    full_H[i,j] = np.full((s), v_H[i, i])
                    # print(np.full((100), v_H[i, i]))
                else:
                    # print('v_H is a vector of size ')
                    # print(v_H[i, j].shape)                                  # array of values
                    full_H[i, j] = v_H[i, j].reshape(1,s)

    # return v_W, v_dW, v_H, v_sigma
    return v_W, v_dW, full_H, v_sigma



def SinglePoint(_W, l, _point):
    # print("Evaluate a single point")

    v = _W.evalf(subs={l: _point})

    if v == 0:
        v = 0.0

    return v