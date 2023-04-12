from skimage import io, morphology, filters
from skimage.color import rgb2gray, rgba2rgb
from skimage.morphology import label
from skimage.segmentation import clear_border
import matplotlib.pyplot as plt
import numpy as np
# plt.rcParams['image.cmap'] = 'gray'

def biasField(I,mask):
    (rows,cols) = I.shape
    r, c = np.meshgrid(list(range(rows)), list(range(cols)))
    rMsk = r[mask].flatten()
    cMsk = c[mask].flatten()
    VanderMondeMsk = np.array([rMsk*0+1, rMsk, cMsk, rMsk**2, rMsk*cMsk, cMsk**2]).T
    ValsMsk = I[mask].flatten()
    coeff, residuals, rank, singularValues = np.linalg.lstsq(VanderMondeMsk, ValsMsk, rcond=-1)
    VanderMonde = np.array([r*0+1, r, c, r**2, r*c, c**2]).T
    J = np.dot(VanderMonde, coeff) # @ operator is a python 3.5 feature!
    J = J.reshape((rows,cols)).T
    return(J)


def biasCorrect(image="images/1carr-96etoh-alexa-sted-decon.tif",
                    threshold=None):
    # Image readin
    im = rgb2gray(rgba2rgb(io.imread(image)))

    # Noise reduction by median filtering
    dskelm = morphology.disk(1)
    imFilt = filters.median(im, dskelm)

    # Thresholding
    tProteins = threshold if threshold != None else filters.threshold_otsu(imFilt)
    proteins = imFilt > tProteins

    # Bias correction
    B = biasField(imFilt, proteins)
    imBias = imFilt - B + B.mean()

    # Finding new threshold
    tProteinsBias = threshold if threshold != None else filters.threshold_otsu(imBias)
    proteinsBias = imBias > tProteinsBias

    # Clearing borders
    clearProteinsBias = clear_border(proteinsBias)

    # Labelling
    labels = label(clearProteinsBias)

    return labels