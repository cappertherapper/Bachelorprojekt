import numpy as np
import math
import matplotlib.pyplot as plt
import ripleys as rp

N = 100 # Number of points
M = 256 # M+1 is the width and height of the domain
L = 50 # Maximum distances to consider
lowerBound = [L, L]
upperBound = [M-L, M-L]


rand_data = np.random.uniform(0, M, (N, 2))

plt.scatter(rand_data[:,0], rand_data[:,1])
plt.show()

r, K = rp.ripleysPlot(rand_data, lowerBound, upperBound)
K = K[r<L]
r = r[r<L]

xs = np.linspace(0, L, 100)

plt.plot(xs, math.pi*xs**2)
plt.plot(r, K)
plt.show()