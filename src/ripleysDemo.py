import numpy as np
import matplotlib
from skimage.measure import regionprops
from matplotlib import pyplot as plt,patches
from scipy.spatial.distance import pdist, squareform
# def ripleysK1 (K,r,y):
from preprocessing import process_image


def poissonProcess(N, M):  
    # N randomly and uniformly distributed points on the domain 0..Mx0..M
    x = M * np.random.rand(N, 2)
    return x


def noisyGrid(N, s, M):
    # ceil(sqrt(N))^2 points sampled on a regular grid 0..Mx0..M grid plus gaussian noise with standard deviation s in 2D
    rows, cols = np.meshgrid(np.linspace(0, M, int(np.ceil(np.sqrt(N)))),
                             np.linspace(0, M, int(np.ceil(np.sqrt(N)))))
    x = np.vstack([rows.flatten(), cols.flatten()]).T + s*np.random.randn(int(np.ceil(np.sqrt(N)))**2, 2)
    x = x[(x >= 0).all(axis=1) & (x <= M).all(axis=1), :]
    return x


def regularGrid(N, M):
    # ceil(sqrt(N))^2 points sampled on a regular grid 0..Mx0..M grid in 2D
    rows, cols = np.meshgrid(np.linspace(0, M, int(np.ceil(np.sqrt(N)))),
                             np.linspace(0, M, int(np.ceil(np.sqrt(N)))))
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
    # N = 100
    # M = 256
    N = 542 # Same number of clusters found in .tif
    M = 378 # Size of .tif images
    L = 50
    x0 = np.array([L,L])
    x1 = np.array([M-L,M-L])
    # x0 = [L,L]
    # x1 = [M-L,M-L]
    F = 1
    
    DATA = [5,3]
    REPEAT = [1,2]
    COLOR = ['r','b']
    THICKNESS = [1,3]
    PAUSELEN = 0.5
    
    IMAGE_PATH = "images/1carr-96etoh-alexa-sted-decon.tif"
    IMAGE_THRESHOLD = 0.5

    
    fig, ax = plt.subplots(1,3, figsize=(10,6), gridspec_kw={'width_ratios': [1, 1, 3]})  # plots
    fig.subplots_adjust(wspace=0.4)
    plt.tight_layout()
    for i in range(len(REPEAT)):
        for j in range(REPEAT[i]):
            ax[i].clear()
            if DATA[i] == 1:
                # A poisson process
                x = poissonProcess(N, M)
                name = 'Poisson'
            elif DATA[i] == 2:
                # Mother process of Gaussians
                x = motherOfGaussians(30, M//50, N, M) 
                name = 'Mother of Gaussian'
            elif DATA[i] == 3:
                # A regular grid plus noise
                x = noisyGrid(N, 0.2*M/np.sqrt(N), M)
                name = 'Noisy Grid'
            elif DATA[i] == 4:
                # A regular grid
                x = regularGrid(N, M)  
                name = 'Grid'
            elif DATA[i] == 5:
                # Centroids of thresholded clusters
                y = process_image(IMAGE_PATH, IMAGE_THRESHOLD)
                regions = regionprops(y)
                x = np.array([x.centroid for x in regions])
                name = 'Centroids'
            elif DATA[i] == 6:
                # Thresholded clusters
                y = process_image(IMAGE_PATH, IMAGE_THRESHOLD)
                y = np.rot90(y, axes=(1,0))
                bw = y > IMAGE_THRESHOLD
                x = np.argwhere(bw == True)
                print(x.shape)
                name = 'White Pixels'
                
            K1, r1, y = ripleysK1(x, x0, x1)
            r = np.linspace(0, L, 10*L)
            f = np.pi*r**2
            
            ax[i].plot(x[:, 0], x[:, 1], 'b+')
            rect = patches.Rectangle(x0, (x1[0]-x0[0]),(x1[0]-x0[0]), color='red')
            ax[i].add_patch(rect)
            ax[i].axis(xmin=0,xmax=M)
            ax[i].axis(ymin=0,ymax=M)
            ax[i].set_title(name)
            ax[2].plot(r1, K1, COLOR[i], linewidth=THICKNESS[0])
            ax[2].plot(r, f, 'k')
            ax[2].axis(xmin=0,xmax=L)
            ax[2].axis(ymin=0,ymax=8000)
            ax[2].set_title("Ripley's K")
            plt.pause(PAUSELEN)
    plt.show()

    
ripleysDemo()
