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

# Noise reduction by median filtering
dskelm = morphology.disk(1)
imFilt = filters.median(im, dskelm)

# Thresholding by otsu
tProteins = filters.threshold_otsu(imFilt)
proteins = imFilt > tProteins

# Bias correction
B = biasField(imFilt, proteins)
imBias = imFilt - B + B.mean()

# Finding new threshold
tProteinsBias = filters.threshold_otsu(imBias)
proteinsBias = imBias > tProteinsBias

# Plotting
plt.imshow(proteinsBias)
plt.title('Mask at ' + str(tProteinsBias))
plt.show()
