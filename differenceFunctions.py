import numpy as np
import matplotlib.pyplot as plt

# Different ways of interpreting difference between functions
# Functions are of form [[x0,y0], [x1,y1], ..., [xn, yn]]

def average_diff(f,g):
    return abs(np.average(g - f))

def integral_diff(f,g):
    return abs(np.sum(g - f))

def max_diff(f,g):
    return np.sum(np.absolute(g - f), axis=1).max()

def mse_diff(f, g):
    return np.mean(np.sum((g - f)**2, axis=1))

def corrcoef_diff(f, g):
    mean_f = np.mean(f, axis=0)
    mean_g = np.mean(g, axis=0)

    cov_fg = np.mean((g - mean_g) * (f - mean_f))

    return cov_fg / (np.std(g) * np.std(f))


a = np.array([[1,2], [3,4]])
b = np.array([[2,2], [5,1]])


print(np.array(b) - np.array(a))

# print(average_diff(a,b))
# print(integral_diff(a,b))
# print(max_diff(a,b))
# print(mse_diff(a,b))
# print(corrcoef_diff(a,b))


#Difference between two curves with arbitrary x-values. 

def average_abs_diff(f,g):
    fdiff = f[:,1] - np.interp(f[:,0], g[:,0], g[:,1])
    gdiff = g[:,1] - np.interp(g[:,0], f[:,0], f[:,1])
    
    return np.average(abs(fdiff) + abs(gdiff))/2


def average_abs_diff1(f, g, N):
    xs = np.linspace(min(f[0,0], g[0,0]), max(f[-1,0], g[-1,0]), N)        #min or max?
    diff = np.interp(xs, f[:,0], f[:,1]) - np.interp(xs, g[:,0], g[:,1])
    
    return np.average(abs(diff))


#def area_between_curves(f, g):





arr1 = np.array([[0.5, 3], [1, 5],[2, 7], [3, 9]])
arr2 = np.array([[0.7, 3.2], [1.7, 6],[2.2, 7], [4, 12]])

#print(average_abs_diff1(arr1, arr2, 1000))
#print(average_abs_diff(arr1, arr2))


plt.plot(arr1[:,0], arr1[:,1])
plt.plot(arr2[:,0], arr2[:,1])
plt.show()

