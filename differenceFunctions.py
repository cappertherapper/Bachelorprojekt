import numpy as np

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

