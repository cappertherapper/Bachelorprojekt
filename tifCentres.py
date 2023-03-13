import numpy as np
from PIL import Image
from skimage.morphology import label
from skimage.measure import regionprops

def tifCentroids(path='1carr-96etoh-alexa-sted-decon.tif', threshold_value=130):
    # path = '1carr-96etoh-alexa-sted-decon.tif'
    img = Image.open(path).convert('L')
    image = np.array(img)

    bw = image > threshold_value

    label_image = label(bw)

    regions = regionprops(label_image)

    return np.array([x.centroid for x in regions])

# print(tifCentroids().shape)
# print((256 * np.random.rand(100, 2)).max())


def whitePixels(path='1carr-96etoh-alexa-sted-decon.tif', threshold_value=130):
    img = Image.open(path).convert('L')
    image = np.array(img)
    
    bw = image > threshold_value

    return np.argwhere(bw == True)