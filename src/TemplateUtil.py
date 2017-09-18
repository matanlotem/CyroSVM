import numpy as np
from math import sqrt
from scipy.ndimage import measurements
from scipy import signal

KERNEL_GAUSSIAN = 'GAUSSIAN'

"""
Different Template manipulations are collected here
"""


def shape_to_slices(shape, corner = None):
    if not corner:
        corner = [0]*len(shape)
    assert(len(shape) == len(corner))
    return tuple([slice(corner[i],corner[i]+shape[i]) for i in range(len(shape))])


def get_normalize_template_dm(template):
    """
    normalize the L2 norm of the template density_map
    """
    factor = sqrt(np.sum(np.square(template.density_map)))
    if factor != 0:
        return template.density_map / factor
    else:
        raise Exception("Error normalizing template: L2 norm is zero")


def put_template(tomogram_dm, template_dm, position):
    """
    3D READY
    :param tomogram_dm:  dm is density map
    :param template_dm: dm is density map
    :param position: center of cube/square position
    :return:
    """

    # find corner position of template
    corner = [position[i] - side // 2 for i,side in enumerate(template_dm.shape)]
    # slice matchng positions in template and tomogram
    template_slice = [slice(0, side) for side in template_dm.shape]
    tomogram_slice = [slice(corner[i], corner[i] + side) for i,side in enumerate(template_dm.shape)]

    # if template sticks out of tomogram edges, trim slices so it fits
    for i in range(len((tomogram_slice))):
        if (tomogram_slice[i].start < 0): # tempale starts before tomogram
            template_slice[i] = slice(-tomogram_slice[i].start,template_slice[i].stop)
            tomogram_slice[i] = slice(0,tomogram_slice[i].stop)
        if (tomogram_slice[i].stop > tomogram_dm.shape[i]):  # template ends after tomogram
            template_slice[i] = slice(template_slice[i].start,template_slice[i].stop + tomogram_dm.shape[i] - tomogram_slice[i].stop)
            tomogram_slice[i] = slice(tomogram_slice[i].start, tomogram_dm.shape[i])

    # place template in tomogram
    tomogram_dm[tuple(tomogram_slice)] += template_dm[tuple(template_slice)]


    #shape = tuple([slice(corner[i],corner[i] + template_dm.shape[i]) for i in range(len(corner))])
    #if ([shape[i].start < 0 or shape[i].stop > tomogram_dm.shape[i] for i in range(len(tomogram_dm.shape))].count(True) > 0):
    #    assert(False)
    #tomogram_dm[shape] += template_dm




def align_densitymap_to_COM(densitymap, container_size_3D):
    """
    :param densitymap: the density map to align
    :param container_size_3D: shape of resulting container (tuple of dimension sizes- for 2D use z_size=1 for both)
    :return: a new tomogram whose COM is its center
    """
    #only 2d fit in 3D arrays
    assert(len(densitymap.shape) == len(container_size_3D) and len(densitymap.shape) == 3 and densitymap.shape[2] == 1 and container_size_3D[2] == 1)
    #all sizes must be odd
    assert ([x%2==1 for x in densitymap.shape].count(False) == 0)
    assert ([x % 2 == 1 for x in container_size_3D].count(False) == 0)


    flat_dm = densitymap[:,:,0]
    container_size_2D = container_size_3D[:2]

    big_container = np.zeros([2*x+1 for x in container_size_2D])
    big_container_center = np.floor(np.array(big_container.shape) / 2)

    densitymap_center = np.floor(np.array(flat_dm.shape) / 2)

    put_template(big_container, flat_dm, big_container_center.astype(int).tolist())

    COM = np.array(measurements.center_of_mass(big_container))

    top_left_corner = (np.floor(COM - ((np.array(container_size_2D) - 1)/2))).astype(int).tolist()
    shape = shape_to_slices(container_size_2D, top_left_corner)
    truncated_matrix = big_container[shape]

    return truncated_matrix.reshape(container_size_3D)


def create_kernel(name, dim, gaussian_size, gaussian_stdev):
    """
    Creats a kernel of the specified kind and dimension.
    :param name: Kind of kernel to create. Only KERNEL_GAUSSIAN at the moment.
    :param dim: Dimension of the kernel. Only 2 of 3.
    :param gaussian_size: size of gaussian
    :param gaussian_stdev: standard deviation of gaussian
    :return: 3 dimensional ndarray where the third dimension is of size 1 for the 2D case.
    """
    if KERNEL_GAUSSIAN == name:
        base = signal.gaussian(gaussian_size, gaussian_stdev)
        if 2 == dim:
            return np.outer(base, base).reshape(len(base), len(base), 1)
        elif 3 == dim:
            plane = np.outer(base, base).reshape(len(base), len(base), 1)
            kernel = np.outer(base, plane[0]).reshape(len(base), len(base), 1)
            for row in plane[1:]:
                kernel = np.concatenate((kernel, np.outer(base, row).reshape(len(base), len(base), 1)), 2)
            return kernel
        else:
            raise NotImplementedError('Dimension can\'t be %d! (only 2 or 3)' % dim)
    else:
        raise NotImplementedError('No kernel option %s!' % name)


if __name__ == '__main__':
    pass