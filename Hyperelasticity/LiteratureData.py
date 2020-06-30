import os
import numpy as np
import matplotlib.pyplot as plt

from Hyperelasticity.HyperelasticModel import HyperelasticModel
from Hyperelasticity.utils import Common

from Hyperelasticity.Literature import BrainSurgerySim
from Hyperelasticity.Literature import AbnormalDevelopments
from Hyperelasticity.Literature import PathologicalGrowing
from Hyperelasticity.Literature import MRIRegistration
from Hyperelasticity.Literature import BrainShift
from Hyperelasticity.Literature import MRElastography
from Hyperelasticity.Literature import Hyperelastic


# E = 2*G*(1 + v)


def Load():

    dir = 'C:\\UCL\\PhysicsSimulation\\Python\\Biomechanics\\GaussianProcesses\\data'
    # before: ./data/lambda.npy
    _lambda = np.load(os.path.join(dir,'lambda.npy'))
    L = np.load(os.path.join(dir,'L.npy'))
    W = np.load(os.path.join(dir,'W.npy'))
    dW = np.load(os.path.join(dir,'dW.npy'))
    H = np.load(os.path.join(dir,'H.npy'))
    sigma = np.load(os.path.join(dir,'sigma.npy'))
    Y = np.load(os.path.join(dir,'Y.npy'))
    C = np.load(os.path.join(dir,'C.npy'))
    A = np.load(os.path.join(dir,'A.npy'))
    M = np.load(os.path.join(dir,'M.npy'))

    return _lambda, L, W, dW, H, sigma, Y, C, A, M


def Process():

    """ Generate data """
    # BRAIN SURGERY SIMULATION
    l1, W1, dW1, H1, sigma1, P1, offset1, Y1, C1, A1, M1 = BrainSurgerySim.Process()

    # # SIMULATION OF ABNORMAL DEVELOPMENTS
    l2, W2, dW2, H2, sigma2, P2, offset2, Y2, C2, A2, M2 = AbnormalDevelopments.Process()

    # # LINEAR PATHOLOGICAL GROWING REGION
    l3, W3, dW3, H3, sigma3, P3, offset3, Y3, C3, A3, M3 = PathologicalGrowing.Process()

    # # LINEAR MRI REGISTRATION
    l4, W4, dW4, H4, sigma4, P4, offset4, Y4, C4, A4, M4 = MRIRegistration.Process()

    # LINEAR BRAIN SHIFT COMPENSATION FOR ABLATION
    l5, W5, dW5, H5, sigma5, P5, offset5, Y5, C5, A5, M5 = BrainShift.Process()

    # # LINEAR MRE
    l6, W6, dW6, H6, sigma6, P6, offset6, Y6, C6, A6, M6 = MRElastography.Process()

    # # NON-LINEAR
    l7, W7, dW7, H7, sigma7, P7, offset7, Y7, C7, A7, M7 = Hyperelastic.Process()

    input('break')



    """ Join data """
    L = l1
    L = np.concatenate((L, l2), axis=0)
    L = np.concatenate((L, l3), axis=0)
    L = np.concatenate((L, l4), axis=0)
    L = np.concatenate((L, l5), axis=0)
    L = np.concatenate((L, l6), axis=0)
    L = np.concatenate((L, l7), axis=0)
    W = W1
    W = np.concatenate((W, W2), axis=0)
    W = np.concatenate((W, W3), axis=0)
    W = np.concatenate((W, W4), axis=0)
    W = np.concatenate((W, W5), axis=0)
    W = np.concatenate((W, W6), axis=0)
    W = np.concatenate((W, W7), axis=0)
    dW = dW1
    dW = np.concatenate((dW, dW2), axis=0)
    dW = np.concatenate((dW, dW3), axis=0)
    dW = np.concatenate((dW, dW4), axis=0)
    dW = np.concatenate((dW, dW5), axis=0)
    dW = np.concatenate((dW, dW6), axis=0)
    dW = np.concatenate((dW, dW7), axis=0)
    H = H1
    H = np.concatenate((H, H2), axis=0)
    H = np.concatenate((H, H3), axis=0)
    H = np.concatenate((H, H4), axis=0)
    H = np.concatenate((H, H5), axis=0)
    H = np.concatenate((H, H6), axis=0)
    H = np.concatenate((H, H7), axis=0)
    sigma = sigma1
    sigma = np.concatenate((sigma, sigma2), axis=0)
    sigma = np.concatenate((sigma, sigma3), axis=0)
    sigma = np.concatenate((sigma, sigma4), axis=0)
    sigma = np.concatenate((sigma, sigma5), axis=0)
    sigma = np.concatenate((sigma, sigma6), axis=0)
    sigma = np.concatenate((sigma, sigma7), axis=0)
    P = P1
    P = np.concatenate((P, P2), axis=0)
    P = np.concatenate((P, P3), axis=0)
    P = np.concatenate((P, P4), axis=0)
    P = np.concatenate((P, P5), axis=0)
    P = np.concatenate((P, P6), axis=0)
    P = np.concatenate((P, P7), axis=0)
    O = offset1 + offset2 + offset3 + offset4 + offset5 + offset6 + offset7
    O = np.asarray(O)
    Y = Y1 + Y2 + Y3 + Y4 + Y5 + Y6 + Y7
    Y = np.asarray(Y)
    C = C1 + C2 + C3 + C4 + C5 + C6 + C7
    C = np.asarray(C)
    A = A1 + A2 + A3 + A4 + A5 + A6 + A7
    A = np.asarray(A)
    M = M1 + M2 + M3 + M4 + M5 + M6 + M7
    M = np.asarray(M)

    print('Data')
    print(L.shape)
    print(W.shape)
    print(dW.shape)
    print(H.shape)
    print(sigma.shape)
    print(P.shape)
    print(O.shape)
    print(Y.shape)
    print(C.shape)
    print(A.shape)
    print(M.shape)





    """ Plot """
    r = Common.lambdaRange
    model = HyperelasticModel('general')
    _lambda = model.stretches
    palette = ['#d4d6bd', '#34bac5', '#008486', '#6caf7f', '#b5d568', '#fcaf29', '#e16a69']
    colourcat = ['#34bac5', '#5e80bc', '#7061db',
                 '#b5d568', '#6caf7f', '#008486',
                 '#d4d6bd', '#fcaf29', '#e16a69']


    fig = plt.figure(1)
    fig.suptitle("Brain soft tissue characterisation", fontsize=14)

    axis = plt.subplot(141)
    p1 = plt.plot(np.transpose(l1), np.transpose(sigma1), c=palette[0])
    p2 = plt.plot(np.transpose(l2), np.transpose(sigma2), c=palette[1])
    p3 = plt.plot(np.transpose(l3), np.transpose(sigma3), c=palette[2])
    p4 = plt.plot(np.transpose(l4), np.transpose(sigma4), c=palette[3])
    p5 = plt.plot(np.transpose(l5), np.transpose(sigma5), c=palette[4])
    p6 = plt.plot(np.transpose(l6), np.transpose(sigma6), c=palette[5])
    p7 = plt.plot(np.transpose(l7), np.transpose(sigma7), c=palette[6])
    plt.title("Cauchy stress (sigma)", fontsize=10)
    plt.xlabel("Stretches (lambda)")
    plt.ylabel("Cauchy stress (sigma)")
    axis.set_xlim([r[0], r[1]])
    axis.set_ylim([0, 4e7])

    axis = plt.subplot(142)
    p1 = plt.plot(np.transpose(l1), np.transpose(W1), c=palette[0])
    p2 = plt.plot(np.transpose(l2), np.transpose(W2), c=palette[1])
    p3 = plt.plot(np.transpose(l3), np.transpose(W3), c=palette[2])
    p4 = plt.plot(np.transpose(l4), np.transpose(W4), c=palette[3])
    p5 = plt.plot(np.transpose(l5), np.transpose(W5), c=palette[4])
    p6 = plt.plot(np.transpose(l6), np.transpose(W6), c=palette[5])
    p7 = plt.plot(np.transpose(l7), np.transpose(W7), c=palette[6])
    plt.title("Strain energy density function", fontsize=10)
    plt.xlabel("Stretches (lambda)")
    plt.ylabel("Strain energy (W)")
    # axis.set_xlim([r[0], r[1]])
    # axis.set_ylim([0, 4e7])
    axis.set_xlim([0.6, 1.4])
    axis.set_ylim([0, max(map(max, W))])

    axis = plt.subplot(143)
    p1 = plt.plot(np.transpose(l1), np.transpose(dW1), c=palette[0])
    p2 = plt.plot(np.transpose(l2), np.transpose(dW2), c=palette[1])
    p3 = plt.plot(np.transpose(l3), np.transpose(dW3), c=palette[2])
    p4 = plt.plot(np.transpose(l4), np.transpose(dW4), c=palette[3])
    p5 = plt.plot(np.transpose(l5), np.transpose(dW5), c=palette[4])
    p6 = plt.plot(np.transpose(l6), np.transpose(dW6), c=palette[5])
    p7 = plt.plot(np.transpose(l7), np.transpose(dW7), c=palette[6])
    plt.title("Energy Gradient", fontsize=10)
    plt.xlabel("Stretches (lambda)")
    plt.ylabel("Energy Gradien (dW)")
    axis.set_xlim([r[0], r[1]])
    axis.set_ylim([-4e7, 4e7])

    axis = plt.subplot(144)
    p1 = plt.plot(np.transpose(l1), np.transpose(H1), c=palette[0])
    p2 = plt.plot(np.transpose(l2), np.transpose(H2), c=palette[1])
    p3 = plt.plot(np.transpose(l3), np.transpose(H3), c=palette[2])
    p4 = plt.plot(np.transpose(l4), np.transpose(H4), c=palette[3])
    p5 = plt.plot(np.transpose(l5), np.transpose(H5), c=palette[4])
    p6 = plt.plot(np.transpose(l6), np.transpose(H6), c=palette[5])
    p7 = plt.plot(np.transpose(l7), np.transpose(H7), c=palette[6])
    plt.title("Energy Hessian", fontsize=10)
    plt.xlabel("Stretches (lambda)")
    plt.ylabel("Energy Hessian (H)")
    axis.set_xlim([r[0], r[1]])
    axis.set_ylim([0, 4e7])

    fig.legend((p1[0], p2[0], p3[0], p4[0], p5[0], p6[0], p7[0]),
               ('Surg Sim', 'Abnormal', 'Patholog Grow', 'MRI reg', 'Brain shift', 'MRElastography', 'Hyperelastic'),
               scatterpoints=1,
               loc='lower center',
               ncol=7,
               fontsize=8)

    # plt.tight_layout()
    mng = plt.get_current_fig_manager()
    mng.window.state('zoomed')
    plt.show()

    # save
    np.save('./data/lambda.npy', _lambda)
    np.save('./data/L.npy', L)
    np.save('./data/W.npy', W)
    np.save('./data/dW.npy', dW)
    np.save('./data/H.npy', H)
    np.save('./data/sigma.npy', sigma)
    np.save('./data/Y.npy', Y)
    np.save('./data/C.npy', C)
    np.save('./data/A.npy', A)
    np.save('./data/M.npy', M)

    return _lambda, L, W, dW, H, sigma, Y, C, A, M