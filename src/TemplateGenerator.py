from CommonDataTypes import TiltedTemplate, EulerAngle
from Configuration import CONFIG
from TemplateUtil import align_densitymap_to_COM
from StringComperableEnum import StringComperableEnum
from TemplatesDataAccessObject import BidimensionalLazyFileDAO

import numpy as np
import scipy.ndimage.interpolation
import pickle
import os, shutil, subprocess, time


class TemplatesType(StringComperableEnum):
    GEOMETRIC_3D = 'GEOMETRIC_3D'
    PDBS_3D = 'PDBS_3D'
    ALL_3D = 'ALL_3D'

"""
This file contains all the methods that are used to generate and load templates (from files)
 important methods:
 load_templates_3d(templates_path): load 3d tempaltes produced by chimera bsed code
 generate_tilted_templates_2d(angle_res): create 2d geometric shaped templates
"""

# -------------------------------------------------------------------------------- #
# --------------------------------------- 3D ------------------------------------- #
# -------------------------------------------------------------------------------- #

#The non zero density must lie inside a centered sphere of radius TEMPLATE_DIMENSION/2 so that rotations do not exceed the template size

def fill_with_sphere(dm, rad):
    for x in range(2*rad):
        for y in range(2*rad):
            for z in range(2*rad):
                if (x-rad)**2 + (y-rad)**2 + (z-rad)**2 <= rad**2:
                    dm[x,y,z] = 1
    return dm


def generate_templates_3d(output_path, angle_res, templates_type):
    """
    Generates 3D tilted templates density maps by calling chimera script.
    Path to chimera.exe (chimera instalation) and to ChimeraUtils (part of project)
    must be specified in the configuration.

    :param output_path: directory where output templates and meta data are saved.
           Wipes content, creates if does not exist
    :param angle_res: discrete resolution for tilts generated
    :param templates_type: supports GEOMETRIC_3D, PDBS_3D, ALL_3D

    :return tilted_templates object - refer to load_tempaltes_3d
    """

    # check output directory exists and clean it, if doesn't exist then create it
    if output_path[-1] != '\\':
        output_path += '\\'
    if os.path.isdir(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path)


    # prepare command
    script_name = '.\chimera_template_generator.py'
    cmnd = format(r'"%s" --debug --nogui --script "%s -o %s -a %d'%(CONFIG.CHIMERA_PATH,
                                                                    script_name,
                                                                    os.path.abspath(output_path),
                                                                    angle_res))
    if templates_type in (TemplatesType.GEOMETRIC_3D, TemplatesType.ALL_3D):
        cmnd += ' -g ' + r'.\geometric.txt'
    if templates_type in (TemplatesType.PDBS_3D, TemplatesType.ALL_3D):
        cmnd += ' -p ' + r'.\pdbs.txt'
    if not TemplatesType.contains(templates_type):
        raise Exception('Template generation failed - unknown templates type') # unknown template group
    cmnd += '"'

    # run chimera process
    print('Running command:\n\t' + cmnd)
    subprocess.Popen(cmnd, cwd = CONFIG.CHIMERA_UTILS_PATH)

    # wait while files are being created
    time.sleep(5)
    prev_numfiles = 0
    numfiles = len(os.listdir(output_path))
    while numfiles > prev_numfiles: # as long as files are created, wait
        prev_numfiles = numfiles
        time.sleep(2)
        numfiles = len(os.listdir(output_path))
    if numfiles <= 0:
        raise Exception(format('Template generation failed - %d files generated' % numfiles))

    # load generated templates
    tilted_templates = load_templates_3d(output_path)
    return tilted_templates


def load_templates_3d(templates_path):
    """
    Load templates as created by chimera based template_generator
    :param templates_path: path containing output of chimera template generation
    :return: BidimensionalLazyFileDAO containing templates where the first index corresponds
             to template_id and the second index corresponds to tilt_id
    """

    # load metadata pickles
    template_metadata = pickle.load(open(templates_path + 'template_ids.p','rb'))
    tilt_metadata = pickle.load(open(templates_path + 'tilt_ids.p','rb'))
    EulerAngle.init_tilts_from_list(tilt_metadata )

    # load and create tilted template for every tilt_id and template_id
    tilted_templates = BidimensionalLazyFileDAO(templates_path, len(template_metadata) , len(tilt_metadata))
    return tilted_templates


# -------------------------------------------------------------------------------- #
# --------------------------------------- 2D ------------------------------------- #
# -------------------------------------------------------------------------------- #

def fill_with_square(dm, side):
    dim = dm.shape[0]
    top_left = int(dim/2-side/2)
    for x in range(int(side)):
        for y in range(int(side)):
            dm[top_left+x,top_left+y] = 1
    return dm


def fill_with_circle(dm, rad):
    dim = dm.shape[0]
    for x in range(dim):
        for y in range(dim):
            if (x-dim/2)**2 + (y-dim/2)**2 <= rad**2:
                dm[x,y] = 1
    return dm

#   |--w1--|
# _   ____
# |  |x __| h2
# h1 | |
# |  | |
# _  |_|
#     w2
#   x is the position of the center of the picture
#
def fill_with_L(dm, h1, w1, h2, w2, flip= False):
    dim = dm.shape[0]
    corner_x = (dim - 1)/2
    corner_y = (dim - 1)/2
    offset_x = (w2 - 1)/2
    offset_y = (h2 - 1)/2
    #I want the center of rotation to be the corner
    ind_map = lambda p: (int(corner_y - p[1]), int(corner_x + p[0]))
    if flip:
        ind_map = lambda p: (int(corner_y - p[1]), int(corner_x - p[0]))
    for x in range(int(w1)):
        for y in range(int(h1)):
            dm[ind_map((x - offset_x, y - offset_y))] = 1

    for x in range(int(w2), int(w1)):
        for y in range(int(h2), int(h1)):
            dm[ind_map((x - offset_x, y - offset_y))] = 0
    return dm


# checks if the inequality:
# 0 <= p1 + p2 < bound
# holds for each index
def check_if_in_bound(bound, p1, p2):
    for i in range(len(bound)):
        if bound[i] <= p1[i] + p2[i] or p1[i] + p2[i] < 0:
            return False
    return True


def fill_with_rand_shape(dm, dim, n_iterations=10, blur=True):
    for i in range(n_iterations):
        add_random_shape(dm)
    if blur:
        import Noise
        dm = Noise.blur_filter(dm, dim)
    return dm


def add_random_shape(dm):
    s = dm.shape
    p = (rnd(s[0]), rnd(s[1]), rnd(s[2])) #pick a random point

    rnd_dim = rnd(2, 5)  #dim must be odd
    sub_shape = (2*rnd_dim + 1, 2*rnd_dim + 1, 2*rnd_dim + 1)
    sub_map = np.zeros(sub_shape)
    if rnd(2) == 0:
        sub_map = np.ones(sub_shape)  #sub_shape is a cube
    else:
        sub_map = fill_with_sphere(sub_map, rnd_dim - 1)
    #sub_map = np.ones((rnd_dim, rnd_dim, rnd_dim))
    for x in range(-rnd_dim, rnd_dim):
        for y in range(-rnd_dim, rnd_dim):
            for z in range(-rnd_dim, rnd_dim):
                if check_if_in_bound(s, p,(x, y, z)):
                    dm[x + p[0], y + p[1], z + p[2]] += sub_map[rnd_dim - x, rnd_dim - y, rnd_dim -z]
                    #dm[x + p[0], y + p[1], z + p[2]] += sub_map[x,y,z]


def rotate2d(dm, theta):
    rotated = scipy.ndimage.interpolation.rotate(dm, theta)
    rotated_dim = rotated.shape[0]
    original_dim = dm.shape[0]

    return rotated[rotated_dim//2-original_dim//2:rotated_dim//2-original_dim//2 + original_dim,rotated_dim//2-original_dim//2:rotated_dim//2-original_dim//2 + original_dim]


def generate_tilted_templates_2d(angle_res):
    """
    :param angle_res: discrete resolution for tilts generated

    :return: tuple of tuples of TiltedTemplates (each group has the same template_id)
    """

    template_dimensions_2d = tuple([CONFIG.TEMPLATE_DIMENSION, CONFIG.TEMPLATE_DIMENSION,1])
    # create different template density maps
    # circle
    circle_dm = np.zeros(template_dimensions_2d )
    fill_with_circle(circle_dm[:, :, 0], CONFIG.TEMPLATE_DIMENSION // 4)
    # square
    square_dm = np.zeros(template_dimensions_2d)
    fill_with_square(square_dm[:, :, 0], CONFIG.TEMPLATE_DIMENSION // 2)
    # L
    L_dm = np.zeros(template_dimensions_2d)
    fill_with_L(L_dm[:, :, 0], CONFIG.TEMPLATE_DIMENSION // 2, CONFIG.TEMPLATE_DIMENSION // 3, 3, 3)
    # flipped L
    flipped_L_dm = np.zeros(template_dimensions_2d)
    fill_with_L(flipped_L_dm[:, :, 0], CONFIG.TEMPLATE_DIMENSION // 2, CONFIG.TEMPLATE_DIMENSION // 3, 3, 3, True)

    # create tilts
    EulerAngle.init_tilts_from_list([(phi, 0, 0) for phi in range(0, 360, angle_res)])

    # create tilted templates tuple of tuples
    templates = []
    for template_id, template_dm in enumerate([circle_dm, square_dm, L_dm, flipped_L_dm]):
        specific_templates = []
        for tilt_id , euler_angle in enumerate(EulerAngle.Tilts):
            specific_templates.append(TiltedTemplate(align_densitymap_to_COM(rotate2d(template_dm, euler_angle.Phi), template_dimensions_2d), tilt_id, template_id))
        templates.append(tuple(specific_templates))
    return tuple(templates)


# -------------------------------------------------------------------------------- #
# ------------------------------------ GENERAL ----------------------------------- #
# -------------------------------------------------------------------------------- #


def generate_tilted_templates():  # TODO: remove
    """
    :return: tuple of tuples of TiltedTemplates (each group has the same template_id)
    """
    return generate_tilted_templates_2d()


# -------------------------------------------------- Generators ------------------------------------------------------ #
# Enum containing all the supported generators
class TemplateGenerator(StringComperableEnum):
    LOAD = 'LOAD'
    LOAD_3D = 'LOAD_3D'
    SOLID = 'SOLID'
    CHIMERA = 'CHIMERA'


# TODO: Place holders for template generator
def template_loader(paths):
    if not False:
        for path in paths:
            with open(path, 'rb') as file:
                yield pickle.load(file)
    else:
        for path in paths:
            with open(path, 'wb') as file:
                yield lambda x: pickle.dump(x, file)


def generate_chimera(paths, angular_resolution, template_type):
    generate_templates_3d(paths[0], angular_resolution, template_type)
    return load_templates_3d(paths[0])


def template_generator_solid(paths):
    templates = generate_tilted_templates()
    for i, path in enumerate(paths):
        template = templates[i]
        with open(path, 'wb') as file:
            pickle.dump(template, file)
        yield template


def template_generator_solid_2d(paths):
    for template in generate_tilted_templates_2d():
        yield template


def template_generator_3d_load(paths):
    load_templates_3d(paths[0])


if __name__ == '__main__':
    pass