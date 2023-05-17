from skimage import filters
from skimage.morphology import label
import numpy as np


def generate_noise(size, threshold, smooth):
    """Generates a random noise image"""   

    im = np.random.normal(size=(size, size))
    imFilt = filters.gaussian(im, smooth)
    imThresh = imFilt > threshold

    return label(imThresh)



def generate_noise_array(size, threshold, smooth, amount=2):
    """Generates a list of random noise images"""   
    
    noise_list = []
    for i in range(amount):
        im = np.random.normal(size=(size, size))
        imFilt = filters.gaussian(im, smooth)
        imThresh = imFilt > threshold
        noise_list.append(label(imThresh))

    return np.array(noise_list)


# def generate_noise_tensor(size, threshold, disk_size, amount=1):
#     """Generates a list of random noise images"""   
    
#     noise_list = []
#     for i in range(amount):
#         im = np.random.normal(size=(size, size))

#         disk = morphology.disk(disk_size)
#         imFilt = filters.median(im, disk)

#         thresh = threshold if threshold != None else filters.threshold_otsu(imFilt)
#         imThresh = imFilt > thresh

#         labels = label(imThresh)
#         noise_list.append(labels)

#     return np.array(noise_list)