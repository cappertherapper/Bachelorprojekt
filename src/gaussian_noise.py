from skimage import morphology, filters
from skimage.morphology import label
import numpy as np


def generate_noise(size, threshold, disk_size, amount=1):
    """Generates a list of random noise images"""   
    
    noise_list = []
    for i in range(amount):
        im = np.random.normal(size=(size, size))
        disk = morphology.disk(disk_size)
        imFilt = filters.median(im, disk)
        thresh = threshold if threshold != None else filters.threshold_otsu(imFilt)
        
        imThresh = imFilt > thresh

        labels = label(imThresh)
        noise_list.append(labels)

    return np.array(noise_list)

def generate_noise_avg(size, threshold, disk_size, amount=1):
    """Generates a list of random noise images"""   
    
    noise_list = []
    for i in range(amount):
        im = np.random.normal(size=(size, size))

        disk = morphology.disk(disk_size)
        imFilt = filters.median(im, disk)
        noise_list.append(imFilt)

    imFilt = np.mean(noise_list,axis=0)
    print(imFilt)
    thresh = threshold if threshold != None else filters.threshold_otsu(imFilt)
    imThresh = imFilt > thresh
    print("thres",imThresh)

    labels = label(imThresh)
    # noise_list.append(labels)

    return labels