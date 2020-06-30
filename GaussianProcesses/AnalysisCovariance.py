

import numpy as np
import scipy
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm
import scipy.stats as st
import math
from scipy.spatial.distance import cdist
import GPy

sns.set(color_codes=True)


''' Load GP model '''
def load_GP():
    # params = np.load('./data/great/GP_icm12_params.npy')
    # X = np.load('./data/great/GP_icm12_X.npy')
    # y = np.load('./data/great/GP_icm12_y.npy')
    # n = np.load('./data/great/GP_icm12_n.npy')

    # params = np.load('C:/UCL/PhysicsSimulation/Python/Biomechanics/GaussianProcesses/data/great/GP_icm12_params.npy', allow_pickle=True)
    # X = np.load('C:/UCL/PhysicsSimulation/Python/Biomechanics/GaussianProcesses/data/great/GP_icm12_X.npy', allow_pickle=True)
    # y = np.load('C:/UCL/PhysicsSimulation/Python/Biomechanics/GaussianProcesses/data/great/GP_icm12_y.npy', allow_pickle=True)
    # n = np.load('C:/UCL/PhysicsSimulation/Python/Biomechanics/GaussianProcesses/data/great/GP_icm12_n.npy', allow_pickle=True)

    params = np.load('C:/UCL/PhysicsSimulation/Python/Biomechanics/GaussianProcesses/data/GP_icm12_params.npy', allow_pickle=True)
    X = np.load('C:/UCL/PhysicsSimulation/Python/Biomechanics/GaussianProcesses/data/GP_icm12_X.npy', allow_pickle=True)
    y = np.load('C:/UCL/PhysicsSimulation/Python/Biomechanics/GaussianProcesses/data/GP_icm12_y.npy', allow_pickle=True)
    n = np.load('C:/UCL/PhysicsSimulation/Python/Biomechanics/GaussianProcesses/data/GP_icm12_n.npy', allow_pickle=True)

    _r = 1
    K = GPy.kern.Matern32(1)
    icm = GPy.util.multioutput.ICM(input_dim=1, num_outputs=12, kernel=K)

    m = GPy.models.GPCoregionalizedRegression(X, y, kernel=icm)

    m.update_model(False)
    m.initialize_parameter()
    m[:] = params
    m.update_model(True)
    # print("m=", m)

    return m, X, y, n


def compute_covariance_matrix(m):
    # compute covariance matrix
    W = m.ICM.B.W.values
    k = m.ICM.B.kappa.values
    B = W * W.transpose() + k * np.eye(12)

    return B


def plot_covariance(B):
    # data

    N = B.shape[0]
    # mean, cov = [0, 1], [[1, .5], [.5, 1]]
    mean, cov = np.zeros(N), B
    data = np.random.multivariate_normal(mean, cov, 200)
    vars = ['t'+str(t) for t in range(1,N+1,1)]
    # df = pd.DataFrame(data, columns=["x", "y"])
    df = pd.DataFrame(data, columns=vars)

    g = sns.jointplot(x="t10", y="t11", data=df, kind="kde", color="m")
    g.plot_joint(plt.scatter, c="w", s=30, linewidth=1, marker="+")
    g.ax_joint.collections[0].set_alpha(0)
    g.set_axis_labels("$t10$", "$t11$")
    plt.show()

    g = sns.jointplot(x="t9", y="t11", data=df, kind="kde", color="m")
    g.plot_joint(plt.scatter, c="w", s=30, linewidth=1, marker="+")
    g.ax_joint.collections[0].set_alpha(0)
    g.set_axis_labels("$t9$", "$t11$")
    plt.show()

    g = sns.PairGrid(df)
    g.map_diag(sns.kdeplot)
    g.map_offdiag(sns.kdeplot, n_levels=6)
    plt.show()


def compute_kernel_gaussian(centre=(.0,.0), image_size=(10,10), sigma=1):
    # https://stackoverflow.com/questions/29731726/how-to-calculate-a-gaussian-kernel-matrix-efficiently-in-numpy
    x_axis = np.linspace(0, image_size[0] - 1, image_size[0]) - centre[0]
    y_axis = np.linspace(0, image_size[1] - 1, image_size[1]) - centre[1]
    xx, yy = np.meshgrid(x_axis, y_axis)
    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sigma))

    return kernel

# def compute_kernel_rbf()

# def gkern(kernlen=21, nsig=3):
#     """Returns a 2D Gaussian kernel."""
#
#     x = np.linspace(-nsig, nsig, kernlen+1)
#     kern1d = np.diff(st.norm.cdf(x))
#     kern2d = np.outer(kern1d, kern1d)
#     return kern2d/kern2d.sum()

def plot_sample(k):
    xx, yy = np.mgrid[-3:3:30j, -3:3:30j]
    X = np.vstack((xx.flatten(), yy.flatten())).T
    K = k.K(X)
    s = np.random.multivariate_normal(np.zeros(X.shape[0]), K)
    #plt.contourf(xx, yy, s.reshape(*xx.shape), cmap=plt.cm.hot)
    plt.imshow(s.reshape(*xx.shape), interpolation='nearest')
    plt.colorbar()
    plt.show()



m, X, y, n = load_GP()
B = compute_covariance_matrix(m)
plot_covariance(B)

# https://matplotlib.org/examples/color/colormaps_reference.html
kernel = compute_kernel_gaussian(centre=(25, 40), image_size=(100, 50), sigma=10)
# kernel = gkern()
plt.imshow(kernel, cmap=cm.viridis)
plt.show()
print("max at :", np.unravel_index(kernel.argmax(), kernel.shape))
print("kernel shape", kernel.shape)


ker1 = GPy.kern.RBF(1)
ker2 = GPy.kern.RBF(input_dim=1, variance = .65, lengthscale=2.)
plot_sample(ker2)


# https://www.aidanscannell.com/post/gaussian-process-regression/
def kernel_prior_rbf(x1, x2, var_f, l):
    """Squared exponential kernel"""
    #return var_f * np.exp((-cdist(x1, x2)**2) / (2*l**2))
    n = x1.shape[0]
    dist = np.asarray([float(np.sqrt((t1-t2)**2)) for t1 in x1 for t2 in x2]).reshape(n,n)
    return var_f * np.exp((-dist**2) / (2*l**2))

def kernel_prior_periodic(x1, x2, var_f, l, p):
    """Periodic kernel"""
    n = x1.shape[0]
    dist = np.asarray([float(np.abs(t1 - t2)) for t1 in x1 for t2 in x2]).reshape(n, n)
    return var_f * np.exp(-(2*np.sin(np.pi*dist/p)**2) / (l**2))

def kernel_prior_linear(x1, x2, var_f, var_b, c):
    """Linear kernel"""
    n = x1.shape[0]
    dist = np.asarray([float(t1-c)*float(t2-c) for t1 in x1 for t2 in x2]).reshape(n, n)
    return var_b + var_f * dist

def kernel_prior_matern32(x1, x2, var_f, l):
    """Linear kernel"""
    n = x1.shape[0]
    dist = np.asarray([float(np.sqrt((t1 - t2) ** 2)) for t1 in x1 for t2 in x2]).reshape(n, n)
    return var_f * (1+np.sqrt(3)*dist/l) * np.exp(-np.sqrt(3)*dist/l)

n = 25  # number of test points
x_min = -5
x_max = 5
x_star = np.linspace(x_min*1.4, x_max*1.4, n).reshape(-1,1) # points we're going to make predictions at

l = 2 # lengthscale hyper-parameter
var_f = 0.65  # signal variance hyper-parameter
rbf_K = kernel_prior_rbf(x_star, x_star, var_f, l)  # prior covariance
plt.imshow(rbf_K, cmap=cm.viridis)
plt.show()

variance = 0.65
length = 1.
periodicity = 0.5
periodic_K = kernel_prior_periodic(x_star, x_star, variance, length, periodicity)  # prior covariance
plt.imshow(periodic_K, cmap=cm.viridis)
plt.show()

variance = 0.27
variance_b = 0.47
offset = -2.
linear_K = kernel_prior_linear(x_star, x_star, variance, variance_b, offset)  # prior covariance
plt.imshow(linear_K, cmap=cm.viridis)
plt.show()

variance = 2.
length = 20.
matern32_K = kernel_prior_matern32(x_star, x_star, variance, length)  # prior covariance
plt.imshow(matern32_K, cmap=cm.viridis)
plt.show()