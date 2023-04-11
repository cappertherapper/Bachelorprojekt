import numpy as np
import matplotlib.pyplot as plt
from skimage.draw import polygon2mask
from scipy import ndimage

# t = np.linspace(0,2*np.pi,11)   

# canvas = np.zeros((1024,1024))
# a = 0.5
# xs = np.array([np.cos(t[0:-2:2]), a*np.cos(t[1:-1:2])])
# ys = np.array([np.sin(t[0:-2:2]), a*np.sin(t[1:-1:2])])
# star = np.array([np.concatenate(xs.T), np.concatenate(ys.T)]).T

# t = np.linspace(0,2*np.pi,1000)
# circle = np.array([np.cos(t), np.sin(t)]).T

# star_mask = polygon2mask((1024,1024), (star*40) + 100)
# circle_mask = polygon2mask((1024,1024), (circle*100) + 500)
# all_mask = star_mask + circle_mask


from PIL import Image
from skimage.morphology import label


path = 'images/1carr-96etoh-alexa-sted-decon.tif'
img = Image.open(path).convert("L")
image = np.array(img)
threshold_value = 130
bw = image > threshold_value
label_image = label(bw)

canvas = np.zeros((378,378))

fig, ax = plt.subplots(1,4, figsize=(14,6), gridspec_kw={'width_ratios': [1, 1, 1, 2]})
fig.subplots_adjust(wspace=0.4)
plt.tight_layout()
ax[3].imshow(label_image)


F = np.zeros(300)
G = np.zeros(300)
for cluster in range (1, label_image.max()+1):
    ref_cluster = label_image == cluster
    rem_clusters = np.logical_and((label_image != cluster), (label_image != 0))


    D = ndimage.distance_transform_edt(ref_cluster==0)
    f = np.zeros(300)
    g = np.zeros(300)
    for i in range(0,300):
        K = D<=i
        f[i] = np.count_nonzero(K*rem_clusters)
        g[i] = np.count_nonzero(K)
        #if (i%25==0):
            #i = 2
            #plt.contour(K)
            #plt.pause(0.5)
    F += f
    G += g

F = F / label_image.max()
G = G / label_image.max()

ax[0].plot(F)
ax[0].set_xlabel('r')
ax[0].set_ylabel('area overlap')

ax[1].plot(F/G)
ax[1].set_xlabel('r')
ax[1].set_ylabel('fractional area overlap')

ax[2].plot(F[1:]-F[:-1])
ax[2].set_xlabel('r')
ax[2].set_ylabel('curve overlap')
plt.show()