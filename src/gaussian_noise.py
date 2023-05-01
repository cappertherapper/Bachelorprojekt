from skimage import morphology, filters
from skimage.morphology import label
import numpy as np
from torch import randn

def generate_noise(size, amount, threshold, disk_size,torch_method=False):
    """Generates a list of random noise images"""   
    
    noise_list = []
    for i in range(amount):
        if torch_method==True:
           im=randn(size,size)
        else:
            im = np.random.normal(size=(size, size))

        disk = morphology.disk(disk_size)
        imFilt = filters.median(im, disk)

        thresh = threshold if threshold != None else filters.threshold_otsu(imFilt)
        imThresh = imFilt > thresh

        labels = label(imThresh)
        noise_list.append(labels)

    return np.array(noise_list)
