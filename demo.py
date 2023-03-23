import numpy as np
import matplotlib.pyplot as plt
from skimage.draw import polygon2mask
from scipy import ndimage

t = np.linspace(0,2*np.pi,11)   

canvas = np.zeros((1024,1024))
a = 0.5
xs = np.array([np.cos(t[0:-2:2]), a*np.cos(t[1:-1:2])])
ys = np.array([np.sin(t[0:-2:2]), a*np.sin(t[1:-1:2])])
star = np.array([np.concatenate(xs.T), np.concatenate(ys.T)]).T

t = np.linspace(0,2*np.pi,1000)
circle = np.array([np.cos(t), np.sin(t)]).T

star_mask = polygon2mask((1024,1024), (star*40) + 100)
circle_mask = polygon2mask((1024,1024), (circle*100) + 500)
all_mask = star_mask + circle_mask

plt.imshow(all_mask)

D = ndimage.distance_transform_edt(circle_mask==0)
f = np.zeros(600)
g = np.zeros(600)
for i in range(0,600):
    K = D<=i
    f[i] = np.count_nonzero(K*star_mask)
    g[i] = np.count_nonzero(K)
    if (i%50==0):
        i = 2
        plt.contour(K)
        plt.pause(0.5)

plt.show()
    
plt.plot(f)
plt.xlabel('r')
plt.ylabel('area overlap')
plt.show()

plt.plot(f/g)
plt.xlabel('r')
plt.ylabel('fractional area overlap')
plt.show()

plt.plot(f[1:]-f[:-1])
plt.xlabel('r')
plt.ylabel('curve overlap')
plt.show()