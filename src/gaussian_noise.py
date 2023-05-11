from skimage import morphology, filters
from skimage.morphology import label
import numpy as np
import torch

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

def generate_noise_tensor(size, threshold, disk_size, amount=1):
    """Generates a list of random noise images"""   
    
    noise_list = []
    for i in range(amount):
        im = torch.randn(size, size)

        disk = morphology.disk(disk_size)
        disk = torch.from_numpy(disk.astype(float)).to(torch.float32).unsqueeze(0).unsqueeze(0)
        imFilt = filters.median(torch.unsqueeze(im, 0), disk)

        thresh = threshold if threshold is not None else filters.threshold_otsu(imFilt.numpy())
        thresh = torch.from_numpy(thresh.astype(float)).to(torch.float32).unsqueeze(0).unsqueeze(0)
        imThresh = imFilt > thresh

        labels = label(imThresh)
        noise_list.append(labels)

    return torch.from_numpy(np.array(noise_list)).to(torch.int64)
