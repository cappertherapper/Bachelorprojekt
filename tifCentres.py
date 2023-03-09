import numpy as np
from PIL import Image
from skimage.morphology import label
from skimage.measure import regionprops

def tif_centres(threshold_value=130):
    path = '1carr-96etoh-alexa-sted-decon.tif'
    img = Image.open(path).convert('L')
    image = np.array(img)

    threshold_value = 130
    bw = image > threshold_value

    label_image = label(bw)

    regions = regionprops(label_image)

    return [x.centroid for x in regions]

# print(len(tif_centres(threshold_value=130)))