from skimage import io, morphology, filters
from skimage.color import rgb2gray, rgba2rgb
from skimage.morphology import label
from skimage.segmentation import clear_border
from math import ceil
import numpy as np
import cv2
# plt.rcParams['image.cmap'] = 'gray'

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

def preprocess(image, threshold):
    # Noise reduction by median filtering
    dskelm = morphology.disk(1)
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
    # Image readin
    im = rgb2gray(rgba2rgb(io.imread(path)))
    return preprocess(im, threshold)


def process_video(path, threshold=None, skip_size=1):
    video = cv2.VideoCapture(path)

    length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))

    frames = np.zeros(shape=(ceil(length / skip_size), height, width), dtype=np.uint8)
    frame_count = 0

    while True:
        ret, frame = video.read()

        if not ret:
            break

        if frame_count % skip_size == 0:
            im = rgb2gray(frame)
            im = preprocess(im, threshold)
            frames[ceil(frame_count / skip_size)] = im
        
        frame_count += 1

    video.release()

    return frames
