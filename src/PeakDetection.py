import numpy as np
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion


# Credit: https://stackoverflow.com/questions/3684484/peak-detection-in-a-2d-array
def detect_peaks(image, rank=2, connectivity=2):
    """
    Takes an image and detect the peaks using the local maximum filter.
    Returns a boolean mask of the peaks (i.e. 1 when
    the pixel's value is the neighborhood maximum, 0 otherwise)
    """

    # define a neighborhood
    neighborhood = generate_binary_structure(rank, connectivity)

    # apply the local maximum filter; all pixel of maximal value
    # in their neighborhood are set to 1
    local_max = maximum_filter(image, footprint=neighborhood) == image
    # local_max is a mask that contains the peaks we are
    # looking for, but also the background.
    # In order to isolate the peaks we must remove the background from the mask.

    # we create the mask of the background
    background = (image == 0)

    # a little technicality: we must erode the background in order to
    # successfully subtract it form local_max, otherwise a line will
    # appear along the background border (artifact of the local maximum filter)
    eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)

    # we obtain the final mask, containing only peaks,
    # by removing the background from the local_max mask (xor operation)
    detected_peaks = local_max ^ eroded_background

    return detected_peaks

if __name__ == '__main__':
    image = np.array([[0, 0, 0, 0, 0],
                      [0, .5, .5, .5, 0],
                      [0, .5, 1, .5, 0],
                      [0, .5, .5, .5, 0],
                      [0, 0, 0, 0, 0]],
                     np.float64)
    peaks = detect_peaks(image, 2)
    print(peaks)

    image = np.array([[1, .5, .3, 0, 0],
                      [.5, .7, .5, .4, 0],
                      [.3, .5, 1, .5, .3],
                      [0, .4, .5, .7, .5],
                      [0, 0, .3, .5, 1]],
                     np.float64)
    peaks = detect_peaks(image, 2)
    print(peaks)

    image3d = np.array([[[0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0]],
                        [[0, 0, 0, 0, 0],
                         [0, .5, .5, .5, 0],
                         [0, .5, .5, .5, 0],
                         [0, .5, .5, .5, 0],
                         [0, 0, 0, 0, 0]],
                        [[0, 0, 0, 0, 0],
                         [0, .5, .5, .5, 0],
                         [0, .5, 1, .5, 0],
                         [0, .5, .5, .5, 0],
                         [0, 0, 0, 0, 0]],
                        [[0, 0, 0, 0, 0],
                         [0, .5, .5, .5, 0],
                         [0, .5, .5, .5, 0],
                         [0, .5, .5, .5, 0],
                         [0, 0, 0, 0, 0]],
                        [[0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0]]],
                       np.float64)
    peaks3d = detect_peaks(image3d, 3, 3)
    print(peaks3d)