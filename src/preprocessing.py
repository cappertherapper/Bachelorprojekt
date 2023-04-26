from skimage import io, morphology, filters
from skimage.color import rgb2gray, rgba2rgb
from skimage.morphology import label
from skimage.segmentation import clear_border
from math import ceil
import numpy as np
import cv2

def biasField(I,mask):
    (rows,cols) = I.shape
    r, c = np.meshgrid(list(range(rows)), list(range(cols)))
    rMsk = r[mask].flatten()
    cMsk = c[mask].flatten()
    VanderMondeMsk = np.array([rMsk*0+1, rMsk, cMsk, rMsk**2, rMsk*cMsk, cMsk**2]).T
    ValsMsk = I[mask].flatten()
    coeff, residuals, rank, singularValues = np.linalg.lstsq(VanderMondeMsk, ValsMsk, rcond=-1)
    VanderMonde = np.array([r*0+1, r, c, r**2, r*c, c**2]).T
    J = np.dot(VanderMonde, coeff) # @ operator is a python 3.5 feature!
    J = J.reshape((rows,cols)).T
    return(J)

# Resizes 2d arrays to be same dimensions
def resize(arr):
    # if arr.ndim == 2:
    min_dim_size = min(arr.shape[0], arr.shape[1])
    return arr[:min_dim_size, :min_dim_size]
    # elif arr.ndim == 3:
    #     min_dim_size = min(arr.shape[1], arr.shape[2])
    #     return arr[:, :min_dim_size, :min_dim_size]

def preprocess(image, threshold, disk_size=1):
    # Noise reduction by median filtering
    dskelm = morphology.disk(disk_size)
    imFilt = filters.median(image, dskelm)

    # Thresholding
    tProteins = threshold if threshold != None else filters.threshold_otsu(imFilt)
    proteins = imFilt > tProteins

    # Bias correction
    B = biasField(imFilt, proteins)
    imBias = imFilt - B + B.mean()

    # Finding new threshold
    tProteinsBias = threshold if threshold != None else filters.threshold_otsu(imBias)
    proteinsBias = imBias > tProteinsBias

    # Clearing borders
    clearProteinsBias = clear_border(proteinsBias)

    # Labelling
    labels = label(clearProteinsBias)
    return labels


def process_image(path, threshold=None):
    im = io.imread(path)
    if im.ndim == 2: 
        pass
    elif im.shape[2] == 3:
        im = rgb2gray(im)
    else:
        im = rgb2gray(rgba2rgb(im))
    # im = rgb2gray(rgba2rgb(io.imread(path)))
    im = resize(im)
    return preprocess(im, threshold)

def get_video(path, threshold=None, skip_size=1):

    video = cv2.VideoCapture(path)
    frames = []
    frame_count = 0

    while True:
        ret, frame = video.read()
        if not ret:
            break
        if frame_count % skip_size == 0:
            im = rgb2gray(frame)
            im = resize(im)
            frames.append(im)
        frame_count += 1
        
    video.release()

    frames_array = np.array(frames)
    
    return frames_array

def process_video(path, threshold=None, skip_size=1):
    # video = cv2.VideoCapture(path)

    # length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    # size = min(int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(video.get(cv2.CAP_PROP_FRAME_WIDTH)))

    # frames = np.zeros(shape=(ceil(length / skip_size), size, size))
    # frame_count = 0

    # while True:
    #     ret, frame = video.read()

    #     if not ret:
    #         break

    #     if frame_count % skip_size == 0:
    #         im = rgb2gray(frame)
    #         img = resize(im)
    #         img = preprocess(img, threshold)
    #         frames[ceil(frame_count / skip_size)] = img
        
    #     frame_count += 1

    # video.release()

    video = cv2.VideoCapture(path)
    frames = []
    frame_count = 0

    while True:
        ret, frame = video.read()
        if not ret:
            break
        if frame_count % skip_size == 0:
            im = rgb2gray(frame)
            im = resize(im)
            im = preprocess(im, threshold)
            frames.append(im)
        frame_count += 1
        

    video.release()

    frames_array = np.array(frames)
    
    return frames_array
