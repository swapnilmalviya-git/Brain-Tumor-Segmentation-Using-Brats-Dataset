import numpy as np
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates
import collections
from IPython.display import Image

"""
	source code From.
	https://gist.github.com/oeway/2e3b989e0343f0884388ed7ed82eb3b0
"""

class ElasticTransform(object):
    """Apply elastic transformation on a numpy.ndarray (H x W x C)
    """

    def __init__(self, alpha, sigma):
        self.alpha = alpha
        self.sigma = sigma

    def __call__(self, image, label):
        if isinstance(self.alpha, collections.Sequence):
            alpha = random_num_generator(self.alpha)
        else:
            alpha = self.alpha
        if isinstance(self.sigma, collections.Sequence):
            sigma = random_num_generator(self.sigma)
        else:
            sigma = self.sigma
        return elastic_transform(image, label, alpha=alpha, sigma=sigma)


def elastic_transform(image, label, alpha=1000, sigma=30, spline_order=1, mode='nearest', random_state=np.random):
    """Elastic deformation of image as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    """
    #assert image.ndim == 3
    image = np.array(image)
    label = np.array(label)
    shape = image.shape[:2]

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1),
                         sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1),
                         sigma, mode="constant", cval=0) * alpha

    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    indices = [np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1))]

    result1 = map_coordinates(image, indices, order=spline_order, mode=mode).reshape(shape)
    result2 = map_coordinates(label, indices, order=spline_order, mode=mode).reshape(shape)
    return Image.fromarray(result1), Image.fromarray(result2)


def random_num_generator(config, random_state=np.random):
    if config[0] == 'uniform':
        ret = random_state.uniform(config[1], config[2], 1)[0]
    elif config[0] == 'lognormal':
        ret = random_state.lognormal(config[1], config[2], 1)[0]
    else:
        print(config)
        raise Exception('unsupported format')
    return ret