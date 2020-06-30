"""
Search spaces of hyperelastic function parameters including:
- Neo-Hookean
- Mooney-Rivlin
- Ogden 1-, 2-, 3-term(s)
"""

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from Hyperelasticity import LiteratureData

from Hyperelasticity.utils import Common
from Hyperelasticity.utils import PlotMaterial
from Hyperelasticity.utils import MechanicalTests
from Hyperelasticity.utils import Postulates
from Hyperelasticity import Linear, Nonlinear

kPa = Common.kPa
p = Common.nP
s = Common.sample
c = 6


class HyperSearchSpace():
    def __init__(self):
        # colours used for plots
        self.colourcat = ['#28e0f2', '#34bac5', '#5e80bc', '#7061db', '#bfed50', '#b5d568', '#6caf7f', '#008486', '#d4d6bd', '#fcaf29', '#e8b3ab', '#e16a69']

        # # create model
        # model = Nonlinear.NeoHookean("Neo-Hookean", 0.5)
        # model.Compute()
        # l, W, dW, H, sigma, P, offset = model.Evaluate(MechanicalTests.Uniaxial, [0.01, 5.])
        # self.plot_energy(model.name, l, W, dW, H)
        #
        # model = Nonlinear.MooneyRivlin("Mooney-Rivlin", 0.05, 0.1)
        # model.Compute()
        # l, W, dW, H, sigma, P, offset = model.Evaluate(MechanicalTests.Uniaxial, [0.01, 5.])
        # self.plot_energy(model.name, l, W, dW, H)
        #
        # model = Nonlinear.Ogden("Ogden 1-term", 1, [-0.35], [-2.])
        # model.Compute()
        # l, W, dW, H, sigma, P, offset = model.Evaluate(MechanicalTests.Uniaxial, [0.01, 5.])
        # self.plot_energy(model.name, l, W, dW, H)
        #
        # model = Nonlinear.Ogden("Ogden 2-terms", 2, [-0.25,-0.4], [-2.,-4.15])
        # model.Compute()
        # l, W, dW, H, sigma, P, offset = model.Evaluate(MechanicalTests.Uniaxial, [0.01, 5.])
        # self.plot_energy(model.name, l, W, dW, H)
        #
        # model = Nonlinear.Ogden("Ogden 3-terms", 3, [4.5, 3.7, -1.1], [-0.7, 1.5, -0.4])
        # model.Compute()
        # l, W, dW, H, sigma, P, offset = model.Evaluate(MechanicalTests.Uniaxial, [0.01, 5.])
        # self.plot_energy(model.name, l, W, dW, H)

    def plot_energy(self, name, l, W, dW, H):
        fig = plt.figure(1)
        fig.suptitle(name, fontsize=14)

        #print('min(H)={} > 0.0 [{}]'.format(np.min(H[0, 0]), np.min(H[0, 0]) > 0.0))

        plt.plot(l, W, color=self.colourcat[0], label='W')
        plt.plot(l, dW[0], color=self.colourcat[1], label='dW')
        plt.plot(l, H[0,0], color=self.colourcat[2], label='H')
        plt.legend()
        plt.ylim([0, 5.])
        plt.ylim([0, 3.5])
        plt.show()

    def plot_space(self, name, included, discarded, range1=None, range2=None):
        fig = plt.figure(1)
        fig.suptitle(name+' parameter space', fontsize=14)

        if included.shape[1] == 2:
            # print('min(H)={} > 0.0 [{}]'.format(np.min(H[0, 0]), np.min(H[0, 0]) > 0.0))
            plt.scatter(included[:,0], included[:,1], s=10, c='green', label='included')
            plt.scatter(discarded[:,0], discarded[:,1], s=10, c='red', label='discarded')
            plt.legend()
            plt.xlim(range1)
            plt.ylim(range1)

        elif included.shape[1] == 4:
            ax = fig.add_subplot(111, projection='3d')
            # img = ax.scatter(included[:,0], included[:,1], included[:,2], c=included[:,3], label='included', cmap='viridis')
            img = ax.scatter(included[:, 0], included[:, 1], included[:, 2], c=included[:, 3], label='included', cmap='RdBu')
            fig.colorbar(img)
            plt.legend()
            ax.set_xlim(range1)
            ax.set_ylim(range1)
            ax.set_zlim(range2)
            ax.set_xlabel('X Label')
            ax.set_ylabel('Y Label')
            ax.set_zlabel('Z Label')

        elif included.shape[1] == 6:
            for i in range(3):
                ax = fig.add_subplot(131 + i, projection='3d')
                img = ax.scatter(included[:, 0], included[:, 1], included[:, 2], c=included[:, 3+i], label='included', cmap='viridis')
                fig.colorbar(img)
                plt.legend()
                ax.set_xlim(range1)
                ax.set_ylim(range1)
                ax.set_zlim(range2)
                ax.set_xlabel('X Label')
                ax.set_ylabel('Y Label')
                ax.set_zlabel('Z Label')

        plt.show()

    def neo_hookean(self, sample=10, plot=False):
        mu_space = np.linspace(-5.0, 5.0, sample)
        param_data = []
        discarded = []

        for mu in mu_space:
            model = Nonlinear.NeoHookean("Neo-Hookean", mu)
            model.Compute()
            l, W, dW, H, sigma, P, offset = model.Evaluate(MechanicalTests.Uniaxial, [0.01, 5.])
            if np.min(H[0, 0]) > 0.0:
                if plot:
                    self.plot_energy(model.name+' mu='+str(format(mu,'.2f')), l, W, dW, H)
                param_data.append(mu)
                print('min(H)={} > 0.0 [mu={}]'.format(np.min(H[0, 0]), mu))
            else:
                discarded.append(mu)

        param_data = np.asarray(param_data)
        discarded = np.asarray(discarded)
        print('mu_data = ', param_data)
        print('discarded = ', discarded)
        return param_data, discarded

    def mooney_rivlin(self, sample=10, range=None, plot=False):
        if range is None:
            range = [-5.0, 5.0]
        c1_space = np.linspace(range[0], range[1], sample)
        c2_space = np.linspace(range[0], range[1], sample)
        param_data = []
        discarded = []

        for c1 in c1_space:
            for c2 in c2_space:
                model = Nonlinear.MooneyRivlin("Mooney-Rivlin", c1, c2)
                model.Compute()
                l, W, dW, H, sigma, P, offset = model.Evaluate(MechanicalTests.Uniaxial, [0.01, 5.])
                if np.min(H[0, 0]) > 0.0:
                    title = model.name + ' c1='+format(c1,'.2f') + ' c2='+format(c2,'.2f')
                    if plot:
                        self.plot_energy(title, l, W, dW, H)
                    param_data.append([c1, c2])
                    print('min(H)={} > 0.0 [c1={}, c2={}]'.format(np.min(H[0, 0]), c1, c2))
                else:
                    discarded.append([c1, c2])

        param_data = np.asarray(param_data)
        discarded = np.asarray(discarded)
        self.plot_space("Mooney-Rivlin", param_data, discarded, range)
        print('param_data = ', param_data)
        print('discarded = ', discarded)
        return param_data, discarded

    def ogden_1(self, sample=10, range=None, plot=False):
        if range is None:
            range = [-5.0, 5.0]
        m1_space = np.linspace(range[0], range[1], sample)
        a1_space = np.linspace(range[0], range[1], sample)
        param_data = []
        discarded = []

        for m in m1_space:
            for a in a1_space:
                model = Nonlinear.Ogden("Ogden 1-term", 1, [m], [a])
                model.Compute()
                l, W, dW, H, sigma, P, offset = model.Evaluate(MechanicalTests.Uniaxial, [0.01, 5.])
                if np.min(H[0, 0]) > 0.0:
                    title = model.name + ' m1='+format(m,'.2f') + ' a1='+format(a,'.2f')
                    if plot:
                        self.plot_energy(title, l, W, dW, H)
                    param_data.append([m, a])
                    print('min(H)={} > 0.0 [m1={}, a1={}]'.format(np.min(H[0, 0]), m, a))
                else:
                    discarded.append([m, a])

        param_data = np.asarray(param_data)
        discarded = np.asarray(discarded)
        self.plot_space("Ogden 1-term", param_data, discarded, range)
        print('param_data = ', param_data)
        print('discarded = ', discarded)
        np.save('./data/params_included_Ogden1.npy', param_data)
        np.save('./data/params_discarded_Ogden1.npy', discarded)
        return param_data, discarded

    def ogden_2(self, sample=10, range1=None, range2=None, plot=False):
        if range1 is None:
            range1 = [-5.0, 5.0]
        if range2 is None:
            range2 = [-5.0, 5.0]
        m_space = np.linspace(range1[0], range1[1], sample)
        a_space = np.linspace(range2[0], range2[1], sample)
        param_data = []
        discarded = []

        for m1 in m_space:
            for m2 in m_space:
                for a1 in a_space:
                    for a2 in a_space:
                        if (m1 == 0. and m2 == 0.) or a1 == 0. or a2 == 0.:
                            # print('m1={} m2={} a1={} a2={}'.format(m1, m2, a1, a2))
                            continue
                        model = Nonlinear.Ogden("Ogden 2-term", 2, [m1, m2], [a1, a2])
                        model.Compute()
                        l, W, dW, H, sigma, P, offset = model.Evaluate(MechanicalTests.Uniaxial, [0.01, 5.])
                        if np.min(H[0, 0]) > 0.0:
                            title = model.name + ' m=['+format(m1,'.2f')+','+format(m2,'.2f') + '] a=['+format(a1,'.2f')+','+format(a2,'.2f')+']'
                            if plot:
                                self.plot_energy(title, l, W, dW, H)
                            param_data.append([m1, m2, a1, a2])
                            print('INCLUDED:: min(H)={} > 0.0 [m1=<{},{}>, a1=<{},{}>]'.format(np.min(H[0, 0]), format(m1,'.2f'), format(m2,'.2f'), format(a1,'.2f'), format(a2,'.2f')))
                        else:
                            print('DISCARDED:: min(H)={} > 0.0 [m1=<{},{}>, a1=<{},{}>]'.format(np.min(H[0, 0]), format(m1,'.2f'), format(m2,'.2f'), format(a1,'.2f'), format(a2,'.2f')))
                            discarded.append([m1, m2, a1, a2])

        param_data = np.asarray(param_data)
        discarded = np.asarray(discarded)
        print('param_data = ', param_data)
        print('discarded = ', discarded)
        np.save('./data/params_included_Ogden2.npy', param_data)
        np.save('./data/params_discarded_Ogden2.npy', discarded)
        self.plot_space("Ogden 2-term", param_data, discarded, range1=range1, range2=range2)
        return param_data, discarded

    def ogden_3(self, sample=10, range1=None, range2=None, plot=False):
        if range1 is None:
            range1 = [-5.0, 5.0]
        if range2 is None:
            range2 = [-5.0, 5.0]
        m_space = np.linspace(range1[0], range1[1], sample)
        a_space = np.linspace(range2[0], range2[1], sample)
        param_data = []
        discarded = []

        for m1 in m_space:
            for m2 in m_space:
                for m3 in m_space:
                    for a1 in a_space:
                        for a2 in a_space:
                            for a3 in a_space:
                                if (m1 == 0. and m2 == 0. and m3 == 0.) or a1 == 0. or a2 == 0. or a3 == 0.:
                                    print('m1={} m2={} m3={} a1={} a2={} a3={}'.format(m1, m2, m3, a1, a2, a3))
                                    continue
                                model = Nonlinear.Ogden("Ogden 3-term", 3, [m1, m2, m3], [a1, a2, a3])
                                model.Compute()
                                l, W, dW, H, sigma, P, offset = model.Evaluate(MechanicalTests.Uniaxial, [0.01, 5.])
                                if np.min(H[0, 0]) > 0.0:
                                    title = model.name + ' m=['+format(m1,'.2f')+','+format(m2,'.2f')+','+format(m3,'.2f') + '] a=['+format(a1,'.2f')+','+format(a2,'.2f')+format(a3,'.2f')+']'
                                    if plot:
                                        self.plot_energy(title, l, W, dW, H)
                                    param_data.append([m1, m2, m3, a1, a2, a3])
                                    print('INCLUDED:: min(H)={} > 0.0 [m=<{},{},{}>, a=<{},{},{}>]'.format(np.min(H[0, 0]), format(m1,'.2f'), format(m2,'.2f'), format(m3,'.2f'), format(a1,'.2f'), format(a2,'.2f'), format(a3,'.2f')))
                                else:
                                    print('DISCARDED:: min(H)={} > 0.0 [m=<{},{},{}>, a=<{},{},{}>]'.format(np.min(H[0, 0]), format(m1,'.2f'), format(m2,'.2f'), format(m3,'.2f'), format(a1,'.2f'), format(a2,'.2f'), format(a3,'.2f')))
                                    discarded.append([m1, m2, m3, a1, a2, a3])

        param_data = np.asarray(param_data)
        discarded = np.asarray(discarded)
        print('param_data = ', param_data)
        print('discarded = ', discarded)
        np.save('./data/params_included_Ogden3.npy', param_data)
        np.save('./data/params_discarded_Ogden3.npy', discarded)
        self.plot_space("Ogden 3-term", param_data, discarded, range1=range1, range2=range2)
        return param_data, discarded


search_space = HyperSearchSpace()
# search_space.neo_hookean(sample=10)
# search_space.mooney_rivlin(sample=50)
# search_space.ogden_1(sample=10, range=[-5., 5.])
# search_space.ogden_2(sample=10, range1=[-5.,5.], range2=[-5.,5.], plot=False)
search_space.ogden_3(sample=10, range1=[-5.,5.], range2=[-5.,5.], plot=False)
