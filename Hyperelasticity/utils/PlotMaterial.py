import numpy as np
import matplotlib.pyplot as plt

from Hyperelasticity.utils import Common

r = Common.lambdaRange

def Functions(_name, _lambda, _W, _dW, _H, _sigma):
    fig = plt.figure(1)
    fig.suptitle(_name, fontsize=14)

    axis = plt.subplot(141)
    plt.plot(_lambda, _sigma[0])
    plt.plot(_lambda, _sigma[1])
    plt.plot(_lambda, _sigma[2])
    plt.title("Cauchy stress (sigma)", fontsize=10)
    plt.xlabel("Stretches (lambda)")
    plt.ylabel("Cauchy stress (sigma)")
    axis.set_xlim([0, r[1]])
    axis.set_ylim([0, 4e7])

    axis = plt.subplot(142)
    plt.plot(_lambda, _W)
    plt.title("Strain energy density function", fontsize=10)
    plt.xlabel("Stretches (lambda)")
    plt.ylabel("Strain energy (W)")
    axis.set_xlim([0, r[1]])
    axis.set_ylim([0, 4e7])

    axis = plt.subplot(143)
    plt.plot(_lambda, _dW[0])
    plt.plot(_lambda, _dW[1])
    plt.plot(_lambda, _dW[2])
    plt.title("Energy Gradient", fontsize=10)
    plt.xlabel("Stretches (lambda)")
    plt.ylabel("Energy Gradien (dW)")
    axis.set_xlim([0, r[1]])
    axis.set_ylim([-4e7, 4e7])

    axis = plt.subplot(144)
    for i in range(3):
        n_ij = _H[i, i].shape
        if not n_ij:
            plt.axhline(_H[i, i])       # one value
        else:
            plt.plot(_lambda, _H[i, i]) # array of values
    plt.title("Energy Hessian", fontsize=10)
    plt.xlabel("Stretches (lambda)")
    plt.ylabel("Energy Hessian (H)")
    axis.set_xlim([0, r[1]])
    axis.set_ylim([0, 4e7])

    # plt.tight_layout()
    mng = plt.get_current_fig_manager()
    mng.window.state('zoomed')
    plt.show()




def Prediction(_name, _lambda, _W0, _W0T, _WhatT, _What):
    fig = plt.figure(1)
    fig.suptitle(_name, fontsize=14)

    axis = plt.subplot(141)
    plt.plot(_lambda, np.transpose(_W0))
    plt.title("W (original)", fontsize=10)
    plt.xlabel("Stretches (lambda)")
    plt.ylabel("Strain energy density function (W)")
    axis.set_xlim([r[0], r[1]])
    axis.set_ylim([0, np.max(_W0)*2.0])

    axis = plt.subplot(142)
    plt.plot(_lambda, np.transpose(_W0T))
    plt.title("minmax log(W+0.0001)", fontsize=10)
    plt.xlabel("Stretches (lambda)")
    plt.ylabel("minmax log(W+0.0001)")
    axis.set_xlim([r[0], r[1]])
    axis.set_ylim([0, 1])

    axis = plt.subplot(143)
    plt.plot(_lambda, np.transpose(_WhatT))
    plt.title("prediction minmax log(W+0.0001)", fontsize=10)
    plt.xlabel("Stretches (lambda)")
    plt.ylabel("minmax log(W+0.0001)")
    axis.set_xlim([r[0], r[1]])
    axis.set_ylim([0, 1])

    axis = plt.subplot(144)
    plt.plot(_lambda, np.transpose(_What))
    plt.title("W (prediction)", fontsize=10)
    plt.xlabel("Stretches (lambda)")
    plt.ylabel("Strain energy density function (W)")
    axis.set_xlim([r[0], r[1]])
    axis.set_ylim([0, np.max(_W0)*2.0])

    # plt.tight_layout()
    mng = plt.get_current_fig_manager()
    mng.window.state('zoomed')
    plt.show()





def Groups(_name, _lambda, _L, _W, _dW, _H, _sigma, _Y):
    fig = plt.figure(1)
    fig.suptitle(_name, fontsize=14)

    axis = plt.subplot(141)
    plt.plot(np.transpose(_L), np.transpose(_sigma))
    plt.title("Cauchy stress (sigma)", fontsize=10)
    plt.xlabel("Stretches (lambda)")
    plt.ylabel("Cauchy stress (sigma)")
    axis.set_xlim([r[0], r[1]])
    axis.set_ylim([0, 4e7])

    axis = plt.subplot(142)
    plt.plot(np.transpose(_L), np.transpose(_W))
    for i in range(_W.shape[0]):
        # plt.text(_lambda[i], _W[i], 'Text')
        plt.text(_L[i,0], _W[i,0], _Y[i])
    plt.title("Strain energy density function", fontsize=10)
    plt.xlabel("Stretches (lambda)")
    plt.ylabel("Strain energy (W)")
    # axis.set_xlim([0.6, 1.4])
    axis.set_xlim([0, 5])
    axis.set_ylim([0, max(map(max, _W))])

    axis = plt.subplot(143)
    plt.plot(np.transpose(_L), np.transpose(_dW))
    plt.title("Energy Gradient", fontsize=10)
    plt.xlabel("Stretches (lambda)")
    plt.ylabel("Energy Gradien (dW)")
    axis.set_xlim([r[0], r[1]])
    axis.set_ylim([-4e7, 4e7])

    axis = plt.subplot(144)
    plt.plot(np.transpose(_L), np.transpose(_H))
    plt.title("Energy Hessian", fontsize=10)
    plt.xlabel("Stretches (lambda)")
    plt.ylabel("Energy Hessian (H)")
    axis.set_xlim([r[0], r[1]])
    axis.set_ylim([0, 4e7])

    # plt.tight_layout()
    mng = plt.get_current_fig_manager()
    mng.window.state('zoomed')
    plt.show()




