import numpy as np
# from tqdm.auto import tqdm
from scipy import ndimage
from skimage.measure import regionprops


def analyse_video(video, L=200):
    """Generates lists of F and G from a video"""
    f_list = []
    g_list = []
    # for p in tqdm(range(len(video))):
    for label_image in video:
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

        f = np.zeros(L+1)
        g = np.zeros(L+1)
        for cluster in range(1, curr_image.max()+1):
            ref_mask = curr_image == cluster
            rem_mask = np.logical_and(label_image,np.invert(ref_mask))
            
            D = ndimage.distance_transform_edt(ref_mask==0)
            for i in range(0,L+1):
                dist_mask = D <= i
                f[i] += np.count_nonzero(np.logical_and(dist_mask, rem_mask))
                g[i] += np.count_nonzero(dist_mask)

        f = f / curr_image.max()
        g = g / curr_image.max()

        f_list.append(f)
        g_list.append(g)

    return f_list, g_list

def analyse_image(label_image, L=200):
    """Generates F and G from an image"""

    M = label_image.shape[0] - L    #assumes label_image is square-shaped

    bounded_labels = np.zeros(label_image.shape, dtype=int)
    cluster_num = 1
    for region in regionprops(label_image):
        minr, minc, maxr, maxc = region.bbox
        if (minr > L+1) and (minc > L+1) and (maxr < M-1) and (maxc < M-1):
            pixel_coordinates = region.coords
            bounded_labels[pixel_coordinates[:,0],  pixel_coordinates[:,1]] =  cluster_num
            cluster_num += 1

    f = np.zeros(L+1)
    g = np.zeros(L+1)
    for cluster in range(1, bounded_labels.max()+1):
        ref_mask = bounded_labels == cluster
        rem_mask = np.logical_and(label_image,np.invert(ref_mask))
        
        D = ndimage.distance_transform_edt(ref_mask==0)
        for i in range(0,L+1):
            dist_mask = D <= i
            f[i] += np.count_nonzero(np.logical_and(dist_mask, rem_mask))
            g[i] += np.count_nonzero(dist_mask)

    f = f / bounded_labels.max()
    g = g / bounded_labels.max()

    return f, g



def stochastic_analyse_video(video, L=200):
    """Generates lists of F and G from a video"""
    res_list = []
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

        f = np.zeros(10)
        g = np.zeros(10)
        rand_arr = np.random.randint(1, L+1, size=10)
        for cluster in range(1, curr_image.max()+1):
            ref_mask = curr_image == cluster
            rem_mask = np.logical_and(label_image,np.invert(ref_mask))
            
            D = ndimage.distance_transform_edt(ref_mask==0)
            for i,r in enumerate(rand_arr):
                dist_mask = D <= r
                f[i] += np.count_nonzero(np.logical_and(dist_mask, rem_mask))
                g[i] += np.count_nonzero(dist_mask)

        f = f / curr_image.max()
        g = g / curr_image.max()
        
        res_list.append((f/g, rand_arr))

    return res_list

