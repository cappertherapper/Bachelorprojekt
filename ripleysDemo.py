import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
# def ripleysK1 (K,r,y):

def poissonProcess(N, M):
    # N randomly and uniformly distributed points on the domain 0..Mx0..M
    x = M * np.random.rand(N, 2)
    return x


def noisyGrid(N, s, M):
    # ceil(sqrt(N))^2 points sampled on a regular grid 0..Mx0..M grid plus gaussian noise with standard deviation s in 2D
    rows, cols = np.meshgrid(np.linspace(0, M, int(np.ceil(np.sqrt(N)))) )
    x = np.vstack([rows.flatten(), cols.flatten()]).T + s*np.random.randn(int(np.ceil(np.sqrt(N)))**2, 2)
    x = x[(x >= 0).all(axis=1) & (x <= M).all(axis=1), :]
    return x


def regularGrid(N, M):
    # ceil(sqrt(N))^2 points sampled on a regular grid 0..Mx0..M grid in 2D
    rows, cols = np.meshgrid(np.linspace(0, M, np.ceil(np.sqrt(N))), np.linspace(0, M, np.ceil(np.sqrt(N))))
    x = np.concatenate((rows.reshape(-1, 1), cols.reshape(-1, 1)), axis=1)
    return x


def motherOfGaussians(L, s, N, M):
    # L uniformly distributed mother processes each with maximually N/L normal distributed children with standard deviation s on a 0..Mx0..M domain
    
    m = 2*s+(M-2*s)*np.random.rand(L, 2)
    x = np.empty((0,2))
    n = int(np.ceil(N/L))
    for i in range(m.shape[0]):
        x = np.vstack((x, m[i,:] + s*np.random.randn(n, 2))) # Uniformly distributed Random set of points
    x = x[(x >= 0).all(axis=1) & (x <= M).all(axis=1)]
    return x

def ripleysK1(x, x0, x1):
    # Calculate Ripleys K function for square domains using sorting

    # Ripley's K function is based on the euclidean distance matrix
    D = squareform(pdist(x, 'euclidean'))

    # We only include measurements on points that are inside the domain (but distances outside are included).
    inside = np.all(x > np.tile(x0, (x.shape[0], 1)), axis=1) & np.all(x < np.tile(x1, (x.shape[0], 1)), axis=1)
    N = np.sum(inside)
    y = x[inside, :]
    D = D[inside, :]
    r = np.sort(D.flatten())
    # We remove 0s from the diagonal except one, since we by hand add the point [0,0].
    r = np.delete(r, np.arange(N-1))

    # The K function is now the index in r
    l = N/np.prod(x1-x0)
    K = ((np.arange(len(r))/N)/l).reshape(-1, 1)
    return K, r, y

def ripleysDemo():
    N = 100
    M = 256
    L = 50
    x0 = np.array([[L,L]])
    x1 = np.array([[M-L,M-L]])
    F = 1
    
    DATA = 1
    REPEAT = [10,1]
    COLOR = ['r','b']
    THICKNESS = [1,3]
    PAUSELEN = 0.1
    
    for i in range(len(REPEAT)):
        for j in range(REPEAT[i]):
            if DATA == 1:
                x = poissonProcess(N, M)  # A poisson process
                name = 'Poisson'
            elif DATA == 2:
                x = motherOfGaussians(30, M//50, N, M)  # Mother process of Gaussians
                name = 'Mother of Gaussian'
            elif DATA == 3:
                x = noisyGrid(N, 0.2*M/np.sqrt(N), M)  # A regular grid plus noise
                name = 'Noisy Grid'
            else:
                x = regularGrid(N, M)  # A regular grid
                name = 'Grid'

            K1, r1, y = ripleysK1(x, x0, x1)
            r = np.linspace(0, L, 10*L)
            f = np.pi*r**2

            plt.subplot(1, len(REPEAT)+1, i+1)
            plt.plot(x[:, 0], x[:, 1], 'b+')
            plt.plot([x0[0], x0[0], x1[0], x1[0], x0[0]], [x0[1], x1[1], x1[1], x0[1], x0[1]], 'r-')
            ind = (x[:, 0] > x0[0]) & (x[:, 0] < x1[0]) & (x[:, 1] > x0[1]) & (x[:, 1] < x1[1])
            plt.plot(x[ind, 0], x[ind, 1], 'r+')
            plt.xlim([0, M])
            plt.ylim([0, M])
            plt.title(name)

            plt.subplot(1, len(REPEAT)+1, len(REPEAT)+1)
            plt.plot(r1, K1, COLOR[i], linewidth=THICKNESS[i])
            plt.plot(r, f, 'k')
            plt.xlim([0, L])
            plt.title("Ripley's K")
            plt.pause(PAUSELEN)
        #figure(1)
        #clf
        
ripleysDemo()
    
