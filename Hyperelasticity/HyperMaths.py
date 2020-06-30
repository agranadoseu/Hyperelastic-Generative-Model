import numpy as np
from sympy import *


C1 = symbols('C1')
D1 = symbols('D1')
muL = symbols('muL')
laL = symbols('laL')
k = symbols('k')    # compression resistance factor
C1=muL/2
D1=laL/2

def invariants():
    print('\n\n>>>>> Invariants-based')

    # variables
    W = symbols('W')
    Ic = symbols('Ic')
    IIc = symbols('IIc')
    IIIc = symbols('IIIc')
    J = symbols('J')
    R = symbols('R')

    # relationships
    J = sqrt(IIIc)

    ''' ENERGY '''
    W = C1*(Ic-3-2*log(J)) + D1*log(J)**2
    R = -(k * (J-1)**3) / 2592      # compression resistance (J<1)

    print('W = ', W)
    print('R = ', R)
    print('W+R = ', W+R)

    ''' GRADIENT '''
    dWdIc = diff(W,Ic)
    dWdIIc = diff(W,IIc)
    dWdIIIc = diff(W,IIIc)
    dW = np.array([dWdIc, dWdIIc, dWdIIIc])

    dRdIc = diff(R, Ic)
    dRdIIc = diff(R, IIc)
    dRdIIIc = diff(R, IIIc)

    print('dW = ', dW)
    print('dW(0) = ', dWdIc)
    print('dW(1) = ', dWdIIc)
    print('dW(2) = ', dWdIIIc)
    print('dR(0) = ', dRdIc)
    print('dR(1) = ', dRdIIc)
    print('dR(2) = ', dRdIIIc)

    print('...... Evaluate [rest] ......')
    _W = W
    _W = _W.subs(muL, 3.44828e+08)
    _W = _W.subs(laL, 2.0*1.55172e+09)    # D1=laL/2
    _W = _W.subs(Ic, 3.0)
    _W = _W.subs(IIIc, 1.0)
    print('     W = ', _W)


def stretches():
    print('\n\n>>>>> Stretches-based')

    # variables
    W = symbols('W')
    l1 = symbols('l1')
    l2 = symbols('l2')
    l3 = symbols('l3')
    J = symbols('J')
    f1 = symbols('f1')
    f2 = symbols('f2')
    f3 = symbols('f3')
    h = symbols('h')
    r = symbols('r')

    ''' ENERGY '''
    f1 = C1 * (l1**2 - 1)
    f2 = C1 * (l2**2 - 1)
    f3 = C1 * (l3**2 - 1)
    h = -muL*log(J) + D1*log(J)**2
    W = f1 + f2 + f3 + h
    r = -(k * (J - 1) ** 3) / 2592  # compression resistance (J<1)

    print('W = ', W)
    print('r = ', r)
    print('W+r = ', W+r)

    ''' GRADIENT '''
    df1dl1 = diff(f1, l1)
    df2dl2 = diff(f2, l2)
    df3dl3 = diff(f3, l3)
    dhdJ = diff(h, J)
    drdJ = diff(r, J)
    dh = diff(h+r, J)

    print('df1 = ', df1dl1)
    print('df2 = ', df2dl2)
    print('df3 = ', df3dl3)
    print('dhdJ = ', dhdJ)
    print('drdJ = ', drdJ)
    print('dh = diff(h+r,J) = ', dh)

    dW1 = df1dl1 + dh * l2 * l3
    dW2 = df2dl2 + dh * l1 * l3
    dW3 = df3dl3 + dh * l1 * l2

    print('dW1 = ', dW1)
    print('dW2 = ', dW2)
    print('dW3 = ', dW3)


invariants()
# stretches()

