from skimage import io, morphology, filters
from skimage.color import rgb2gray, rgba2rgb
from skimage.morphology import label
from skimage.segmentation import clear_border
import numpy as np
import cv2
import matplotlib.pyplot as plt


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


arr = np.zeros((100, 100))
arr[:, 10:20] = 0.1
arr[:, 20:30] = 0.2
arr[:, 30:40] = 0.3
arr[:, 40:50] = 0.4
arr[:, 50:60] = 0.5
arr[:, 60:70] = 0.6
arr[:, 70:80] = 0.7
arr[:, 80:90] = 0.8
arr[:, 90:100] = 0.9

mask=arr>0.3
y=biasField(arr,mask)
plt.imshow(y)