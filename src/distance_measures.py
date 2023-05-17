import numpy as np
# from tqdm.auto import tqdm
from scipy import ndimage
from skimage.measure import regionprops


def analyse_video(video, L=200):
    """Generates lists of F and G from a video"""
    F_list = []
    G_list = []
    # for p in tqdm(range(len(video))):
    for p in range(len(video)):
        label_image = video[p]
        M = label_image.shape[0] - L

        label_image_bounded = np.zeros(label_image.shape, dtype=int)
        cluster_num = 1
        for region in regionprops(label_image):
            minr, minc, maxr, maxc = region.bbox
            if (minr > L+1) and (minc > L+1) and (maxr < M-1) and (maxc < M-1):
                pixel_coordinates = region.coords
                label_image_bounded[pixel_coordinates[:,0],  pixel_coordinates[:,1]] =  cluster_num
                cluster_num += 1

        curr_image = label_image_bounded

        F = np.zeros(L+1)
        G = np.zeros(L+1)
        for cluster in range(1, curr_image.max()+1):
            ref_cluster = curr_image == cluster
            rem_clusters = np.logical_and(label_image,np.invert(ref_cluster))
            
            D = ndimage.distance_transform_edt(ref_cluster==0)
            f = np.zeros(L+1)
            g = np.zeros(L+1)
            for i in range(0,L+1):
                K = D <= i
                f[i] = np.count_nonzero(np.logical_and(K, rem_clusters))
                g[i] = np.count_nonzero(K)
            
            F += f
            G += g 

        F = F / curr_image.max()
        G = G / curr_image.max()

        F_list.append(F)
        G_list.append(G)

    return F_list, G_list

def analyse_image(image, L=200):
    """Generates F and G from an image"""

    label_image = image
    M = label_image.shape[0] - L

    label_image_bounded = np.zeros(label_image.shape, dtype=int)
    cluster_num = 1
    for region in regionprops(label_image):
        minr, minc, maxr, maxc = region.bbox
        if (minr > L+1) and (minc > L+1) and (maxr < M-1) and (maxc < M-1):
            pixel_coordinates = region.coords
            label_image_bounded[pixel_coordinates[:,0],  pixel_coordinates[:,1]] =  cluster_num
            cluster_num += 1

    curr_image = label_image_bounded

    F = np.zeros(L+1)
    G = np.zeros(L+1)
    for cluster in range(1, curr_image.max()+1):
        ref_cluster = curr_image == cluster
        rem_clusters = np.logical_and(label_image,np.invert(ref_cluster))
        
        D = ndimage.distance_transform_edt(ref_cluster==0)
        f = np.zeros(L+1)
        g = np.zeros(L+1)
        for i in range(0,L+1):
            K = D <= i
            f[i] = np.count_nonzero(np.logical_and(K, rem_clusters))
            g[i] = np.count_nonzero(K)
        
        F += f
        G += g 

    F = F / curr_image.max()
    G = G / curr_image.max()

    return F, G