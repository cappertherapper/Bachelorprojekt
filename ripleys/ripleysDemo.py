import numpy as np
import math
import matplotlib.pyplot as plt
import ripleys as rp
import sys
sys.path.append('..')

from tifCentres import tifCentroids



N = 542 # Number of points
M = 378 # M+1 is the width and height of the domain
L = 150 # Maximum distances to consider

lowerBound = [L, L]
upperBound = [M-L, M-L]

centroids = tifCentroids()
#print(centroids)
plt.scatter(centroids[:,0], centroids[:,1])
plt.show()


rand_data = np.random.uniform(0, M, (N, 2))
#print(rand_data)
plt.scatter(rand_data[:,0], rand_data[:,1])
plt.show()

r, K = rp.ripleysPlot(centroids, lowerBound, upperBound)
K = K[r<L]
r = r[r<L]

r1, K1 = rp.ripleysPlot(rand_data, lowerBound, upperBound)
K1 = K1[r1<L]
r1 = r1[r1<L]

xs = np.linspace(0, L, 100)

plt.plot(xs, math.pi*xs**2)
plt.plot(r, K, 'y')
plt.plot(r1, K1)
plt.show()