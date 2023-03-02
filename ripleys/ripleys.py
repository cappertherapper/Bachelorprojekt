import numpy as np
import scipy.spatial.distance as ssd



def ripleysPlot(datapoints, bottomleft, topright):
    D = ssd.squareform(ssd.pdist(datapoints, metric='euclidean'))
    
    inside = np.all((bottomleft < datapoints), 1) & np.all((topright > datapoints), 1)
    
    N = sum(inside)
    D = D[inside,:]
    r = np.sort(D, axis=None)[N:]
    
    A = np.prod(np.subtract(topright, bottomleft))
    K = np.arange(r.size)*A/N**2
    
    return r,K





# def ripleys(datapoints, r, bottomleft, topright):
#     N = datapoints.shape[0]
#     acc = 0
#     for i in range(N):
#         for j in range(N):
#             if (np.linalg.norm(datapoints[i]-datapoints[j]) < r): 
#                 acc+=1
                
#     return (topright-bottomleft)**2/N**2*acc