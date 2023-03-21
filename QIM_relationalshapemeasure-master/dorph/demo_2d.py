import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon
from roipoly import RoiPoly
from matplotlib.patches import Polygon as matlab_poly
from numpy.random import rand
#sets the default font size and line width for all future plots created
#not necessarily needed, as Matlab might have other n
plt.rcParams.update({'font.size': 18, 'lines.linewidth': 3})

#11 values between 0 - 2pi
t = np.linspace(0, 2*np.pi, 11)
a = 0.5
x = np.array([np.cos(t[0:10:2]), a*np.cos(t[1:11:2])])
y = np.array([np.sin(t[0:10:2]), a*np.sin(t[1:11:2])])
# Shape of star (5,4) array = [   [x1,x6,y1,y6],[x2,x7,y2,y7],[...]   ]
star = np.vstack([x, y]).T /2 


# p1: Create a regular polygon with 1000 sides. Shape = (1,1001,2)
# p2: Create a regular polygon with 1000 sides and center at (0.5, 0)
p1 = Polygon([(np.cos(theta), np.sin(theta)) for theta in np.linspace(0, 2*np.pi, 1000)])
p2 = Polygon([(0.5 + np.cos(theta), np.sin(theta)) for theta in np.linspace(0, 2*np.pi, 1000)])

# Subtract p2 from p1 to create a "moon" shape
p3 = p1.difference(p2)

# Extract the vertices of the resulting polygon and divide by 2
moon = np.array(p3.exterior.coords) / 2 #Shape (1003,2)

N = 1024
M = 10
O = 2
I = np.zeros((N, N)) #Probably the grid of points for the canvas?
srm = 32
src = int(N/5)
sr = 64
mr = 128
mx = [1 + mr//2 + rand() * (N-mr-1) for x in range(2)]
print(mx)

#interactively define a region of interest (ROI) in an image
x_data = mx[0]+mr*moon[:,0]
y_data = mx[1]+mr*moon[:,1]
arr= np.array(list(zip(x_data,y_data)))
pol = matlab_poly(arr,edgecolor='r')

fig,ax = plt.subplots()

# im = ax.imshow(I)
# plt.add_patch(pol)
ax.add_patch(pol)
# ax.imshow(I)
# RoiPoly()
# J = np.zeros((N,N))
for i in range(1,O):
    sxm = [1+src/2+sr/2+(N-src-sr-1)*rand() for x in range(2)]
    for j in range(1,M):
        two_rand = [rand() for x in range(2)]
        two_rand = np.array(two_rand)
        sx = src*(two_rand-0.5)+sxm
        x_data = sx[0]+sr*star[:,0]
        y_data = sx[1]+sr*star[:,1]
        indiv_star = matlab_poly(np.array(list(zip(x_data,y_data))),edgecolor='r')
        ax.add_patch(indiv_star)
ax.imshow(I)
RoiPoly()