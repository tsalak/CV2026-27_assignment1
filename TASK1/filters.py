#####TUWIEN - CV: Task1 - Scale-Invariant Blob Detection
#####Group number: 14
from typing import Tuple
import numpy as np
import cv2


def create_log_kernel(size: int, sig: float) -> np.ndarray:
    """
    Returns a rotationally symmetric Laplacian of Gaussian kernel
    with given 'size' and standard deviation 'sig'

    Parameters
    ----------
    size : int
        size of kernel (must be odd) (int)
    sig : int
        standard deviation (float)
    
    Returns
    --------
    float
        kernel: filter kernel (size x size) (float)

    """

    kernel = np.zeros((size, size), np.float64)
    halfsize = int(np.floor(size / 2))
    r = range(-halfsize, halfsize + 1, 1)
    for x in r:
        for y in r:
            hg = (np.power(np.float64(x), 2) + np.power(np.float64(y), 2)) / (2 * np.power(np.float64(sig), 2))
            kernel[x + halfsize, y + halfsize] = -((1.0 - hg) * np.exp(-hg)) / (np.pi * np.power(sig, 4))

    return kernel - np.mean(kernel)


def get_log_pyramid(img: np.ndarray, sigma: float, k: float, levels: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return a LoG scale space of given image 'img' with depth 'levels'
    The filter parameter 'sigma' increases by factor 'k' per level
    HINT: np.multiply(..), cv2.filter2D(..)

    Parameters
    ----------
    img : np.ndarray
        input image (n x m x 1) (float)
    sigma : float
        initial standard deviation for filter kernel
    levels : int
        number of layers of pyramid 

    Returns
    ---------
    np.ndarray
        scale_space : image pyramid (n x m x levels - float)
    np.ndarray
        all_sigmas : standard deviation used for every level (levels x 1 - float)
    """

    # student_code start

    factor = k
    s = sigma

    images = []
    sigmas = []

    for _ in range(levels):
        size = int(2 * np.floor(3*s) + 1)
        kernel = create_log_kernel(size, s) * np.pow(s,2)
        s *= factor 
        images.append(np.abs(cv2.filter2D(img, -1, kernel, borderType=cv2.BORDER_REPLICATE)))
        sigmas.append(s)

    scale_space = np.stack(images)
    all_sigmas = np.array(sigmas)

    # student_code end

    return scale_space, all_sigmas
