# -*- coding: utf-8 -*-
"""
Rewrapping of the Shape Measures Tool
so it does comparisons on 2 binary 3D volumes
"""

import numpy as np
import matplotlib.pyplot as plt
import skimage.measure 
import scipy.ndimage.morphology

from RandomRotations import RandomRotationMatrix
from src import measure_interior, measure_exterior



def binary_volume_interior_measure(ref_volume, 
                                   tar_volume,
                                   marching_cubes_step,
                                   max_d_for_r,
                                   r_step_size,
                                   adaptive_r = True):
    """
    ref_volume & tar_volume: The two binary 3D volumes to be compared
    marching_cubes_step: the step size used for computing triangular meshes.
                         larger step size is faster but also coarser. 
                         if unsure, set to 1.
                         
    adaptive_r: if this is used, max_d_for_r and r_step_size isn't as important     
    
    """

    # computes triangular meshes from 3D binary segmentation mask
    # using the marching cubes algorithm
    
    (verts_tar_volume, 
     faces_tar_volume, 
     normals_tar_volume, 
     values_tar_volume)= skimage.measure.marching_cubes_lewiner(tar_volume, 
                                                                step_size = marching_cubes_step)
    
    # converts array type to uint8 for max compatibility with skimage, scipy and indexing                                                     
    verts_tar_volume_int = verts_tar_volume.astype(np.uint8)

    
    # do a distance transform on the reference array to find the minimum distance
    # of each voxel to a positive voxel. 
    D = scipy.ndimage.morphology.distance_transform_edt(~ref_volume)
    #distances = D[tar_volume.astype(bool)]

    # get the distances for the vertices in the target object triangular mesh
    verts_D = D[verts_tar_volume_int[:,0],
                verts_tar_volume_int[:,1],
                verts_tar_volume_int[:,2]]


    # performs the plots
    r = np.linspace(0, max_d_for_r, r_step_size)

    r_adapt, mu10_adapt, mu11_adapt = measure_exterior(verts_tar_volume, 
                                                       faces_tar_volume, 
                                                       verts_D, 
                                                       r, 
                                                       adaptive = adaptive_r)
    
    
    plt.subplot(1,2,1)
    plt.title('volume')
    plt.plot(r_adapt, mu10_adapt, label='adaptive r')
    plt.legend()

    plt.subplot(1,2,2)
    plt.title('inner surface cut')
    plt.plot(r_adapt, mu11_adapt, label='adaptive r')
    plt.legend()

    plt.show()
    
    
    return r_adapt, mu10_adapt, mu11_adapt 




