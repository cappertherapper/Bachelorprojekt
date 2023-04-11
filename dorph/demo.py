import numpy as np
import matplotlib.pyplot as plt
from skimage.draw import polygon2mask
from scipy import ndimage
from PIL import Image
from skimage.filters import threshold_otsu
from skimage import io, morphology, color
from skimage.morphology import label, closing, square
from skimage.measure import regionprops, regionprops_table
from skimage.color import label2rgb

t = np.linspace(0,2*np.pi,11)   

canvas = np.zeros((1024,1024))
a = 0.5
xs = np.array([np.cos(t[0:-2:2]), a*np.cos(t[1:-1:2])])
ys = np.array([np.sin(t[0:-2:2]), a*np.sin(t[1:-1:2])])
star = np.array([np.concatenate(xs.T), np.concatenate(ys.T)]).T

t = np.linspace(0,2*np.pi,1000)
circle = np.array([np.cos(t), np.sin(t)]).T

path = 'QIM/images/1carr-96etoh-alexa-sted-decon.tif'
img = Image.open(path).convert("L")
image = np.array(img)
threshold_value = 130
bw = image > threshold_value
label_image = label(bw)
# star_mask = polygon2mask((1024,1024), (star*40) + 100)
# circle_mask = polygon2mask((1024,1024), (circle*100) + 500)
F=0
FG=0
FMINUS=0
for i in range(1,(label_image.max()+1)):
    star_mask = (label_image != i) & (label_image != i)
    circle_mask = label_image == i
    # all_mask = star_mask + circle_mask

    # fig, ax = plt.subplots(1,4, figsize=(14,6), gridspec_kw={'width_ratios': [1, 1, 1, 2]})
    # fig.subplots_adjust(wspace=0.4)
    plt.tight_layout()

    # ax[3].imshow(label_image)

    D = ndimage.distance_transform_edt(circle_mask==0)
    f = np.zeros(600)
    g = np.zeros(600)
    for i in range(0,600):
        K = D<=i
        f[i] = np.count_nonzero(K*star_mask)
        g[i] = np.count_nonzero(K)
        if (i%50==0):
            i = 2
            # plt.contour(K)
            # plt.pause(0.5)
    F+=f
    FG+=f/g
    FMINUS+=f[1:]-f[:-1]


fig, ax = plt.subplots(1,4, figsize=(14,6), gridspec_kw={'width_ratios': [1, 1, 1, 2]})
ax[3].imshow(label_image)
fig.subplots_adjust(wspace=0.4)    
ax[0].plot(F/label_image.max())
ax[0].set_xlabel('r')
ax[0].set_ylabel('area overlap')

ax[1].plot(FG/label_image.max())
ax[1].set_xlabel('r')
ax[1].set_ylabel('fractional area overlap')

ax[2].plot(FMINUS/label_image.max())
ax[2].set_xlabel('r')
ax[2].set_ylabel('curve overlap')
plt.show()