from skimage import io, morphology, filters, segmentation
from skimage.color import rgb2gray, rgba2rgb
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['image.cmap'] = 'gray'

def biasField(I,mask):
    (rows,cols) = I.shape
    r, c = np.meshgrid(list(range(rows)), list(range(cols)))
    rMsk = r[mask].flatten()
    cMsk = c[mask].flatten()
    VanderMondeMsk = np.array([rMsk*0+1, rMsk, cMsk, rMsk**2, rMsk*cMsk, cMsk**2]).T
    ValsMsk = I[mask].flatten()
    coeff, residuals, rank, singularValues = np.linalg.lstsq(VanderMondeMsk, ValsMsk)
    VanderMonde = np.array([r*0+1, r, c, r**2, r*c, c**2]).T
    J = np.dot(VanderMonde, coeff) # @ operator is a python 3.5 feature!
    J = J.reshape((rows,cols)).T
    return(J)

# class Preprocessor:
#     def __init__(self, threshold='otsu'):
#         self.threshold = threshold


# Image readin
im = rgb2gray(rgba2rgb(io.imread("1carr-96etoh-alexa-sted-decon.tif")))

# # Noise reduction by median filtering
# dskelm = morphology.disk(1)
# imFilt = filters.median(im, dskelm)

# # Thresholding by otsu
# tProteins = filters.threshold_otsu(imFilt)
# proteins = imFilt > tProteins

# # Bias correction
# B = biasField(imFilt, proteins)
# imBias = imFilt - B + B.mean()

# # Finding new threshold
# tProteinsBias = filters.threshold_otsu(imBias)
# proteinsBias = imBias > tProteinsBias

# # Plotting
# plt.imshow(proteinsBias)
# plt.title('Mask at ' + str(tProteinsBias))
# plt.show()




threshold = filters.threshold_otsu(im)
# background = 0.4


# Noise reduction by median filtering
dskelm = morphology.disk(1)
imFilt = filters.median(im, dskelm)


clusters = imFilt > threshold

# plt.imshow(clusters)
# plt.title('Mask at ' + str(threshold))
# plt.show()

# fig, ax = plt.subplots(1,2)
# ax[0].hist(imFilt.flatten(),100)
# ax[0].plot([threshold,threshold],[0,1000])
# ax[0].set_title('Threshold at ' + str(threshold))
# plt.show()



B = biasField(imFilt, clusters)

print("B (mean, min, max):", B.mean(), B.min(), B.max())
imBias = imFilt-B+B.mean()
fig, ax = plt.subplots(1, 3, figsize=(15,5)) # figsize sets size in inches
ax[0].imshow(imFilt)
ax[0].set_title('Original')
ax[1].imshow(B)
ax[1].set_title('Bias field')
ax[2].imshow(imBias)
ax[2].set_title('Bias field corrected')
plt.show()
 

tClustersBias = filters.threshold_otsu(imBias)
clustersBias = imBias > tClustersBias
plt.imshow(clustersBias)
plt.title('Mask at ' + str(tClustersBias))
plt.show()


fig, ax = plt.subplots(1,2)
ax[0].hist(imFilt.flatten(),100)
ax[0].plot([threshold,threshold],[0,1000])
ax[0].set_title('Threshold at ' + str(threshold))

ax[1].hist(imBias.flatten(),100)
ax[1].plot([tClustersBias,tClustersBias],[0,1000])
ax[1].set_title('Threshold at ' + str(tClustersBias))
plt.show()



# titl = 0.85

# clusters = np.logical_and(titl > im, im > background)

# fig, ax = plt.subplots(1,4)
# ax[0].imshow(im)
# ax[1].imshow(im > threshold)
# ax[2].hist(im.flatten(),100)
# ax[2].plot([threshold, threshold], [0, 5000], 'r-')
# ax[3].imshow(imFilt > threshold)
# plt.title('Histogram and thresholds')
# plt.show()



