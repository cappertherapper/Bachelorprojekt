from skimage import io, morphology, filters
from skimage.color import rgb2gray, rgba2rgb
from skimage.morphology import label
from skimage.segmentation import clear_border
import numpy as np
import cv2


def biasField(I, mask):
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


def resize(arr, size=None, corner=None):
    """Resizes array or ensures that dimensions are the same"""
    start, end = None, None
    min_size = min(arr.shape[0], arr.shape[1])
    if not size:
        start = 0
        end = min_size
    elif not corner:
        start = (min_size - size) // 2
        end = start + size
    else:
        start = corner
        end = corner + size
    return arr[start:end, start:end]


def preprocess(image, threshold, smooth=1, clear_borders=False):
    imFilt = filters.gaussian(image, smooth)

    imThresh = imFilt > threshold

    B = biasField(imFilt, imThresh)
    imBias = imFilt - B + B.mean()

    imBiasThresh = imBias > threshold

    if clear_borders:
        imBiasThresh = clear_border(imBiasThresh)

    labels = label(imBiasThresh)
    return labels


def process_image(path, threshold=None, size=None):
    im = io.imread(path)
    if im.ndim == 2: 
        pass
    elif im.shape[2] == 3:
        im = rgb2gray(im)
    else:
        im = rgb2gray(rgba2rgb(im))
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


def process_video(path, threshold=None, skip_size=1, size=None):
    video = cv2.VideoCapture(path)
    frames = []
    frame_count = 0

    while True:
        ret, frame = video.read()
        if not ret:
            break
        if frame_count % skip_size == 0:
            im = rgb2gray(frame)
            im = resize(im, size)
            im = preprocess(im, threshold)
            frames.append(im)
        frame_count += 1
    
    video.release()

    frames_array = np.array(frames)
    return frames_array
