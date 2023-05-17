import numpy as np
import random
from skimage.measure import label
from src.gaussian_noise import generate_noise, generate_noise_array
import numpy as np


def mean_size(image):
    _, counts = np.unique(label(image), return_counts=True)
    return np.mean(counts[1:])


def find_default_params(image, alpha_range= np.arange(0.5, 3, .1), tau_range= np.arange(0.1, 2, .05)):
    average = mean_size(image)
    im_clusters = image.max()
    # print(average)

    # alpha_range = random.shuffle(alpha_range)
    # tau_range = random.shuffle(tau_range)

    # print(alpha_range)

    best = -9999
    alpha, tau = None, None

    for a in alpha_range:
        for t in tau_range:
            noise = generate_noise(size=image.shape[0], threshold=t, smooth=a)
            noise_clusters = noise.max()
            if noise_clusters:
            # noise_array = generate_noise_array(size=image.shape[0], threshold=t, smooth=a)
                # noise = np.mean(noise_array, axis=0)
            # while not noise.max():
            #     noise = generate_noise(size=image.shape[0], threshold=t, smooth=a)
                curr = mean_size(noise)
                if abs(average - curr) < abs(average - best) and abs(im_clusters - noise_clusters) < im_clusters/2:
                    # print('{:.2f}, {}, {}'.format(curr, a, t))
                    best = curr
                    alpha, tau = a, t
            # print('{}, {}'.format(a, t))
    return alpha, tau


