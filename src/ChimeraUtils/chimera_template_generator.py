import chimera
from VolumeViewer import open_volume_file, volume_from_grid_data
from VolumeData import Array_Grid_Data
from Matrix import euler_xform
import numpy as np
import sys, pickle, argparse



def create_pdb_model(pdb_name, params_list):
    """
    Creates chimera model object from pdb file
    Centers model at center of frame (for future rotations)
    Model object is numbered #1
    
    :param pdb_name: path to pdb file
    :param params_list: sampling resolution
    """
    resoultion = params_list[0]
    # create model from pdb file
    chan = chimera.openModels.open(pdb_name)[0]
    chimera.runCommand('molmap #0 %s modelId 1' % str(resoultion))
    model = chimera.specifier.evalSpec('#1').models()[0]
    chan.destroy()

    # center (for later rotations)
    trans = tuple(np.array(model.openState.cofr.data()) * -1)
    model.openState.globalXform(euler_xform([0,0,0],trans))
    
    return model

def create_cube_matrix(params):
    """
    Creates cube density map matrix - a cube of size side^3
    centered in an empty matirx of size (2*side)^3

    :param params: singleton list - number of pixels in cube edge
    """
    side = params[0]
    matrix = np.zeros((side*2,side*2,side*2))
    matrix[side//2:side//2+side,side//2:side//2+side,side//2:side//2+side] = np.ones((side,side,side))
    return matrix


def create_sphere_matrix(params):
    """
    Creates sphere density map matrix - a sphere of radius rad
    centered in an empty matirx of size (2*rad)^3

    :param params: singleton list- sphere radius in number of pixels
    """
    rad = params[0]
    matrix = np.zeros((2*rad,2*rad,2*rad))
    for x in range(2*rad):
        for y in range(2*rad):
            for z in range(2*rad):
                if (x-rad)**2 + (y-rad)**2 + (z-rad)**2 <= rad**2:
                    matrix[x,y,z] = 1
    return matrix

def shape_to_slices(shape, corner = None):
    if not corner:
        corner = [0]*len(shape)
    assert(len(shape) == len(corner))
    return tuple([slice(corner[i],corner[i]+shape[i]) for i in range(len(shape))])

def create_L_matrix(params):
    """
        |-w1-|
     _   ____
     |  |x __| h2
     h1 | |
     |  | |
     _  |_|
         w2
       x is the the coordinate (0,0,0)

       zdepth is the z axis size
    """
    [w1, w2, h1, h2, zdepth] = params
    assert(w2 <= w1 and h2 <= h1 and zdepth > 0)
    #along the x axis
    x_bar = np.ones((w1,h2,zdepth))
    # along the y axis
    y_bar = np.ones((w2,h1,zdepth))
    matrix = np.zeros((w1, h1, zdepth))
    matrix[shape_to_slices(x_bar.shape)] = x_bar
    matrix[shape_to_slices(y_bar.shape)] = y_bar
    return matrix


def create_geometric_model(shape_name, param_list):
    """
    Creates chimera model object from a geometric shape
    Centers model at center of frame (for future rotations)
    Model object is numbered #1
    
    :param shpae_name: one of known shapes - cube, sphere
    :param param_list: list of geomteric shape paramters (cube side, sphere radius..)
    """
    
    if shape_name == 'cube':
        matrix = create_cube_matrix(param_list)
    elif shape_name == 'sphere':
        matrix = create_sphere_matrix(param_list)
    elif shape_name == 'L':
        matrix = create_L_matrix(param_list)
    else:
        raise('unkown shape!')

    # create model
    v = volume_from_grid_data(Array_Grid_Data(matrix))
    tmp_model = chimera.specifier.evalSpec('#0').models()[0]
    trans = tuple(np.array(tmp_model.openState.cofr.data()) * -1)
    tmp_model.openState.globalXform(euler_xform([0,0,0],trans))

    # change model number to #1 for consitency
    model = tmp_model.copy()
    tmp_model.close()
    return model


def create_model(model_type, model_str, param_list):
    if model_type == 'G':
        return create_geometric_model(model_str, param_list)
    elif model_type == 'P':
        return create_pdb_model(model_str, param_list)
    else:
        raise('unkown model type!')

#reference: scipy source
def center_of_mass(inp, labels=None, index=None):
    normalizer = sum(inp, labels, index)
    grids = np.ogrid[[slice(0, i) for i in inp.shape]]

    results = [sum(inp * grids[dir].astype(float), labels, index) / normalizer for dir in range(inp.ndim)]

    if np.isscalar(results[0]):
        return tuple(results)

    return [tuple(v) for v in np.array(results).T]

def calc_com(matrix):
    """
    Calcultates the center of mass of a density map

    :param matrix: density map matrix
    """
    return np.floor([np.dot(np.array(range(matrix.shape[i])),np.sum(np.sum(matrix,max((i+1)%3,(i+2)%3)),min((i+1)%3,(i+2)%3))) for i in range(3)] / sum(sum(sum(matrix))))

def calc_dim(matrix):
    """
    Calcultates the size of a box that can contain all non zero cells of
    the density map rotated at any angle arround the center of mass

    :param matrix: density map matrix
    """

    # calc radius around center of mass
    rad = 0
    com = calc_com(matrix)
    for x in range(matrix.shape[0]):
        for y in range(matrix.shape[1]):
            for z in range(matrix.shape[2]):
                if matrix[x,y,z] > 0:
                    rad = max(rad, np.sqrt(sum((np.array([x,y,z])-com)**2)))

    # get box dimensions (and add some spares), make it odd, because center is ill defined for even shapes
    dim = int(np.ceil(rad)) * 2 + 3
    return dim



def generate_tilts(angle_res):
    """
    Creates set of all possible euler_angle tilts according to resolution

    :param angle_res: resolution of tilts (for each euler angle)
    """
    
    tilts = []
    for phi in range(0,360,angle_res):
        tilts.append((phi,0,0))
        for theta in range(angle_res, 180, angle_res):
            for psi in range(0, 360, angle_res):
                tilts.append((phi,theta, psi))
        tilts.append((phi,180,0))
        
    return tilts


def create_tilted_model(model, euler_angles):
    """
    Creates tilted model based on original model and euler angles
    Resamples the density map after tilting

    :param model: orignal model to be tilted
    :param euler_angles: 3-tuple of euler angles
    """
    
    # copy
    tmp_model = model.copy()

    # rotate
    rot = euler_xform(euler_angles, [0,0,0])
    tmp_model.openState.globalXform(rot)

    # resample
    chimera.runCommand('vop #0 resample onGrid #1')
    tilted_model = chimera.specifier.evalSpec('#2').models()[0]
    tmp_model.close()
    
    return tilted_model
    

def get_matrix_and_center(matrix, dim):
    """
    Get model density map and place inside a box with side
    of size dim, centered around center of mass

    :param matrix: density map matrix
    :param dim: box side
    """

    # expand matrix
    big_matrix = np.zeros(tuple((2*dim*np.ones((1,3)) + matrix.shape)[0]))
    shape1 = shape_to_slices(matrix.shape, [dim]*3)
    big_matrix[shape1] += matrix

    # center and truncate
    com = [int(x) for x in calc_com(big_matrix)]
    shape2 = shape_to_slices(np.array([dim]*3),[x-dim//2 for x in com])
    truncated_matrix = big_matrix[shape2]
    #if [int(x) for x in calc_com(truncated_matrix)] != [dim//2]*3:
    #    print("COM error: calced_COM:",calc_com(truncated_matrix), " in shape: ", truncated_matrix.shape)
    #    print(com, shape1, shape2, matrix.shape, big_matrix.shape)
    return truncated_matrix


def tilt_and_save(model, euler_angles, dim, output_name):
    """
    Creates and saves a density map of the model tilted according
    to the supplied euler angles

    :param model: orignal model to be tilted
    :param euler_angles: 3-tuple of euler angles
    :param dim: density map box side
    :param output_name: name of density map file
    """
    
    tilted = create_tilted_model(model, euler_angles)
    matrix = get_matrix_and_center(tilted.matrix(), dim)
    print(euler_angles, calc_com(matrix))
    tilted.close()
    np.save(output_name, matrix)


def flow(criteria, angle_res, output_path):
    """
    Creates desnsity map of given models at all possible tilts according
    to given angle resolution and saves to file

    :param criteria: list of template types in the following format:
                     (template_type, template_str, template_var) where
                     - template_type = G: geometric / P: pdb
                     - template_str = shape name / pdb file path
                     - template_var = parameter for model generation (resolution/ radius...)
    :param angle_res: tilt "grid" resolution
    :param output_path: diretory where output is saved
    """
    
    # get dim
    dim = 0
    for criterion in criteria:
        model = create_model(*criterion)
        dim = max(dim,calc_dim(model.matrix()))
        model.close()

    # create tilted density maps
    tilts = generate_tilts(angle_res)
    for template_id, criterion in enumerate(criteria):
        model = create_model(*criterion) # create model
        for tilt_id, tilt in enumerate(tilts): # iterate on tilts
            output_name = output_path + str(template_id) + '_' + str(tilt_id)
            tilt_and_save(model, tilt, dim, output_name)
        model.close() # close model

    # save meta data
    pickle.dump(criteria, open(output_path + 'template_ids.p', 'wb'))
    pickle.dump(tilts, open(output_path + 'tilt_ids.p', 'wb'))


def parse_config(template_type, config_path):
    with open(config_path) as f:
        return [(template_type, line.split('|')[0], [int(param) for param in line.split('|')[1].split(',')]) for line in f.readlines()]

def main(argv):
    # parse arguments
    parser = argparse.ArgumentParser(description = '')
    parser.add_argument('-o', '--output_path', required=True, nargs=1, type=str, help='output dir')
    parser.add_argument('-a', '--angle_res', required=True, nargs=1, type=int, help='angle resolution')
    parser.add_argument('-g', '--geometric_config_path', nargs=1, type=str, help='output dir')
    parser.add_argument('-p', '--pdb_config_path', nargs=1, type=str, help='output dir')
    args = parser.parse_args(argv)
    output_path = args.output_path[0]
    if output_path[-1] != '\\':
        output_path += '\\'

    # set sdtout and stderr
    sys.stdout = open(output_path + 'output.txt', 'w')
    sys.stderr = open(output_path + 'error.txt', 'w')
    print('Start!')

    # get criteria
    criteria = []
    if args.geometric_config_path:
        criteria += parse_config('G', args.geometric_config_path[0])
    if args.pdb_config_path:
        criteria += parse_config('P', args.pdb_config_path[0])
    

    print(criteria)
    print(output_path)
    print(args.angle_res[0])

    # run
    flow(criteria, args.angle_res[0], output_path)

    print('Done!')
    sys.stdout.close()
    sys.stderr.close()
    

if __name__ == '__main__':
    main(sys.argv[1:])
    #tmp = r"C:\Users\guyrom\Documents\GitHub\CryoEmSvm\Chimera\template_generator.py -o C:\Users\guyrom\Documents\GitHub\CryoEmSvm\Chimera\Templates\ -a 60 -g C:\Users\guyrom\Documents\GitHub\CryoEmSvm\Chimera\Templates\geometric.txt"
    #main((tmp.split())[1:])
    # output_path = r'C:\Users\Matan\PycharmProjects\Workshop\Chimera\Templates\\'
    # sys.stdout = open(output_path + 'output.txt', 'w')
    # sys.stderr = open(output_path + 'error.txt', 'w')
    # pdb_name = r'C:\Users\Matan\Dropbox\Study\S-3B\Workshop\Tutotrial\1k4c.pdb'
    # criteria = [('P',pdb_name,10),('G','cube',10),('G','sphere',10)]
    # 
    # flow(criteria, 30, output_path)
    
