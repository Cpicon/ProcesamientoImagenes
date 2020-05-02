""" Joint histogram for color images
    code by Maria Fernanda Roa
    mf.roa@uniandes.edu.co
    IBIO 3470 - Uniandes 
"""
import numpy as np
from skimage.util import img_as_float

def JointColorHistogram(img, num_bins, min_val=None, max_val=None):
    """Calculate joint histogram for color images

    Arguments:
        img (numpy.array) -- 2D color image 
        num_bins (array like of ints) -- Number of bins per channel. 
                                         If an int is given, all channels
                                         will have same ammount of bins.

    Keyword Arguments:
        min_val (array like of ints) -- Minimum intensity range value per channel
                                        If an int is given, all channels
                                        will have same minimmum. (default: {None})
        max_val (array like of ints) -- Maximum intensity range value per channel
                                        If an int is given, all channels
                                        will have same maximum. (default: {None})

    Returns:
        [numpy.array] -- Array containing joint color histogram of size num_bins.
    """   

    assert len(img.shape) == 3, 'img must be a color 2D image'
    #Transform image to float dtype 
    img = img_as_float(img)
    _, _, n_channels = img.shape

    #Verify input parameters
    assert isinstance(num_bins, (int, tuple, list, np.array)),'num_bins must be int or array like'
    if isinstance(num_bins, int):
        num_bins = np.array([num_bins]*n_channels)
    assert len(num_bins) == n_channels,'num_bins length and number of channels differ'
 
    if min_val is None:
        min_val = np.min(img, (0,1))
    else:
        assert isinstance(min_val, (int, tuple, list, np.array)),'min_val must be int or array like'
        if isinstance(min_val, int):
            min_val = np.array([min_val]*n_channels)
        else: 
            min_val = np.array(min_val)
    assert len(min_val) == n_channels,'min_val length and number of channels differ'
    min_val = min_val.reshape((1, 1, -1))

    if max_val is None:
        max_val = np.max(img, (0,1))
    else:
        assert isinstance(max_val, (int, tuple, list, np.array)),'max_val must be int or array like'
        if isinstance(max_val, int):
            max_val = np.array([max_val]*n_channels)
        else: 
            max_val = np.array(max_val)
    assert len(max_val) == n_channels,'max_val length and number of channels differ'
    max_val = max_val.reshape((1, 1, -1)) + 1e-5

    joint_hist = np.zeros(num_bins, dtype=np.int)
    num_bins = num_bins.reshape((1, 1, -1))

    # Scale intensities (intensities are scaled within the range for each channel)
    # Values now are between 0 and 1
    img = (img - min_val) / (max_val - min_val)
    # Calculate index matrix 
    idx_matrix = np.floor(img*num_bins).astype('int') 
    idx_matrix = idx_matrix.reshape((-1, n_channels))
    #Create joint histogram
    for p in range(len(idx_matrix)):
        joint_hist[tuple(idx_matrix[p, :])] += 1

    return joint_hist