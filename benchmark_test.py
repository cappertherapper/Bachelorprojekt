import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from skimage.draw import polygon2mask
from scipy import ndimage
from PIL import Image
from src.preprocessing import process_image, process_video
from skimage.color import label2rgb
from skimage.morphology import label
from IPython.display import display, clear_output
from skimage.measure import regionprops
from concurrent.futures import ThreadPoolExecutor
from matplotlib.colors import ListedColormap
import matplotlib.cm as cm
import timeit

label_video = process_video('images/cheese_gel.avi', threshold=0.3, skip_size=32)


def count_zero_metoden():
    for label_image in label_video:
        L = 100 # Maximum distances to consider
        M = label_image.shape[0] - L

        label_image_bounded = np.zeros(label_image.shape, dtype=int)
        cluster_num = 1
        for region in regionprops(label_image):
            minr, minc, maxr, maxc = region.bbox
            if (minr > L+1) and (minc > L+1) and (maxr < M-1) and (maxc < M-1):
                pixel_coordinates = region.coords
                label_image_bounded[pixel_coordinates[:,0],  pixel_coordinates[:,1]] =  cluster_num
                cluster_num += 1

        bx = (L, M, M, L, L)
        by = (L, L, M, M, L)

        curr_image = label_image_bounded

        F = np.zeros(L+1)
        G = np.zeros(L+1)
        times = []
        for cluster in range(1, curr_image.max()+1):
            ref_cluster = curr_image == cluster
            rem_clusters = label_image * np.invert(ref_cluster)
            D = ndimage.distance_transform_edt(ref_cluster==0)
            f = np.zeros(L+1)
            g = np.zeros(L+1)
            
            K = [D <= x for x in range(L+1)]
            f = np.array([np.count_nonzero(K[i] * rem_clusters) for i in range(L+1)])
            g = np.array([np.count_nonzero(K[i]) for i in range(L+1)])


def loop_metoden():
    for label_image in label_video:
        L = 100 # Maximum distances to consider
        M = label_image.shape[0] - L

        label_image_bounded = np.zeros(label_image.shape, dtype=int)
        cluster_num = 1
        for region in regionprops(label_image):
            minr, minc, maxr, maxc = region.bbox
            if (minr > L+1) and (minc > L+1) and (maxr < M-1) and (maxc < M-1):
                pixel_coordinates = region.coords
                label_image_bounded[pixel_coordinates[:,0],  pixel_coordinates[:,1]] =  cluster_num
                cluster_num += 1

        bx = (L, M, M, L, L)
        by = (L, L, M, M, L)

        curr_image = label_image_bounded

        F = np.zeros(L+1)
        G = np.zeros(L+1)
        times = []
        for cluster in range(1, curr_image.max()+1):
            ref_cluster = curr_image == cluster
            rem_clusters = label_image * np.invert(ref_cluster)
            D = ndimage.distance_transform_edt(ref_cluster==0)
            f = np.zeros(L+1)
            g = np.zeros(L+1)
            
            for i in range(0,L+1):
                K = D <= i
                f[i] = np.count_nonzero(K*rem_clusters)
                g[i] = np.count_nonzero(K)

def np_array():
    for label_image in label_video:
        label_image=label_image[200:700,200:700]
        L = 100 # Maximum distances to consider
        M = label_image.shape[0] - L

        label_image_bounded = np.zeros(label_image.shape, dtype=int)
        cluster_num = 1
        for region in regionprops(label_image):
            minr, minc, maxr, maxc = region.bbox
            if (minr > L+1) and (minc > L+1) and (maxr < M-1) and (maxc < M-1):
                pixel_coordinates = region.coords
                label_image_bounded[pixel_coordinates[:,0],  pixel_coordinates[:,1]] =  cluster_num
                cluster_num += 1

        bx = (L, M, M, L, L)
        by = (L, L, M, M, L)

        curr_image = label_image_bounded

        F = np.zeros(L+1)
        G = np.zeros(L+1)
        times = []
        for cluster in range(1, curr_image.max()+1):
            ref_cluster = curr_image == cluster
            rem_clusters = label_image * np.invert(ref_cluster)
            D = ndimage.distance_transform_edt(ref_cluster==0)
            f = np.zeros(L+1)
            g = np.zeros(L+1)
            
            K = [D <= x for x in range(L+1)]
            f = np.array([np.count_nonzero(K[i] * rem_clusters) for i in range(L+1)])
            g = np.array([np.count_nonzero(K[i]) for i in range(L+1)])


def liste():
    for label_image in label_video:
        label_image=label_image[200:700,200:700]
        
        L = 100 # Maximum distances to consider
        M = label_image.shape[0] - L

        label_image_bounded = np.zeros(label_image.shape, dtype=int)
        cluster_num = 1
        for region in regionprops(label_image):
            minr, minc, maxr, maxc = region.bbox
            if (minr > L+1) and (minc > L+1) and (maxr < M-1) and (maxc < M-1):
                pixel_coordinates = region.coords
                label_image_bounded[pixel_coordinates[:,0],  pixel_coordinates[:,1]] =  cluster_num
                cluster_num += 1

        bx = (L, M, M, L, L)
        by = (L, L, M, M, L)

        curr_image = label_image_bounded

        F = np.zeros(L+1)
        G = np.zeros(L+1)
        times = []
        for cluster in range(1, curr_image.max()+1):
            ref_cluster = curr_image == cluster
            rem_clusters = label_image * np.invert(ref_cluster)
            D = ndimage.distance_transform_edt(ref_cluster==0)
            f = np.zeros(L+1)
            g = np.zeros(L+1)
            
            K = [D <= x for x in range(L+1)]
            f = [np.count_nonzero(K[i] * rem_clusters) for i in range(L+1)]
            g = [np.count_nonzero(K[i]) for i in range(L+1)]


def map_fun():
    for label_image in label_video:
        label_image=label_image[200:700,200:700]
        L = 100 # Maximum distances to consider
        M = label_image.shape[0] - L

        label_image_bounded = np.zeros(label_image.shape, dtype=int)
        cluster_num = 1
        for region in regionprops(label_image):
            minr, minc, maxr, maxc = region.bbox
            if (minr > L+1) and (minc > L+1) and (maxr < M-1) and (maxc < M-1):
                pixel_coordinates = region.coords
                label_image_bounded[pixel_coordinates[:,0],  pixel_coordinates[:,1]] =  cluster_num
                cluster_num += 1

        bx = (L, M, M, L, L)
        by = (L, L, M, M, L)

        curr_image = label_image_bounded

        F = np.zeros(L+1)
        G = np.zeros(L+1)
        times = []
        for cluster in range(1, curr_image.max()+1):
            ref_cluster = curr_image == cluster
            rem_clusters = label_image * np.invert(ref_cluster)
            D = ndimage.distance_transform_edt(ref_cluster==0)
            f = np.zeros(L+1)
            g = np.zeros(L+1)
            
            K = [D <= x for x in range(L+1)]
            f = list(map(lambda k: np.count_nonzero(rem_clusters * k), K))
            g = list(map(lambda k: np.count_nonzero(k), K))



# loop_metoden()
# execution_time = timeit.timeit(loop_metoden, number=20)
# print("Execution time, loop_metoden:", execution_time)

# count_zero_metoden()
# execution_time_2 = timeit.timeit(count_zero_metoden, number=20)
# print("Execution time, count_zero:", execution_time_2)

np_array()
execution_time = timeit.timeit(np_array, number=15)
print("Execution time, np_array:", execution_time)

liste()
execution_time_2 = timeit.timeit(liste, number=15)
print("Execution time, liste:", execution_time_2)


map_fun()
execution_time = timeit.timeit(map_fun, number=15)
print("Execution time, map_fun:", execution_time)