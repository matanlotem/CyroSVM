import numpy as np
import random


class EulerAngle:
    """
    This represents the angles of a rotated rigid body.
    alpha is ...
    """
    Tilts = None

    def __init__(self, Phi, Theta, Psi):
        self.Phi = Phi
        self.Theta = Theta
        self.Psi = Psi

    def __str__(self):
        return str(tuple([self.Phi, self.Theta, self.Psi]))

    @classmethod
    def init_tilts_from_list(cls, tuple_list):
        """

        :param tuple_list: list of 3-tuples containing (phi,theta,psi)
        :return:
        """
        cls.Tilts = [EulerAngle(t[0],t[1],t[2]) for t in tuple_list]

    @classmethod
    def fromTiltId(cls, tilt_id):
        return cls.Tilts[tilt_id]

    @classmethod
    def rand_tilt_id(cls):
        assert(cls.Tilts)
        return random.randint(0, len(cls.Tilts)-1 )



class SixPosition:
    """
    A Combination of center of mass position and a tilt
    """
    def __init__(self, COM_position, tilt_id):
        """
        :param COM_position: 3 tuple
        :param tilt_id:  id that corresponds to EulerAngle's tilt_id
        """
        self.COM_position = COM_position    # 3 tuple
        self.tilt_id = tilt_id              # tilt id

    def __str__(self):
        return str(self.COM_position) + " " + str(self.tilt_id)


class Candidate:
    """
    This is a way to associate metadata to a posistion:
    label, suggested label, and feature vector
    """

    def __init__(self, six_position, suggested_label=None, label = None):
        self.six_position = six_position            # SixPosition
        self.label = label                          # int
        self.suggested_label = suggested_label      # int
        self.features = None                        # list
        self.ground_truth_match = None              # candidate

    @classmethod
    def fromTuple(cls, label, tilt_id, x, y, z = 0):
        return cls(SixPosition((x,y,z), tilt_id), label=label)

    def __str__(self):
        return "Position: " + str(self.six_position) + "\n" +\
               "Suggested Label: " + str(self.suggested_label) + "\n" +\
               "Label: " + str(self.label) + "\n" +\
               "Features: " + str(self.features) + "\n"

    def set_features(self, features):
        self.features = features

    def set_features(self, features):
        self.features = features

    def set_label(self, label):
        self.label = label

    def set_ground_truth(self, ground_truth_candidate):
        self.ground_truth_match = ground_truth_candidate

    def get_tilt_id(self):
        return self.six_position.tilt_id

    def get_position(self):
        return self.six_position.COM_position

    def get_euler_angle(self):
        return EulerAngle.Tilts[self.six_position.tilt_id]


class TiltedTemplate:
    """
    Represents a template that is tilted in a specific euler angle
    contains the density map and the tilt and template metadata
    """
    def __init__(self, density_map, tilt_id, template_id):
        self.template_id = template_id      # int
        self.density_map = density_map      # numpy 3d array
        self.tilt_id = tilt_id              # EulerAngle

    @classmethod
    def fromFile(cls, path, tilt_id, template_id):
        return cls(np.load(path), tilt_id, template_id)


class Tomogram:
    """
    composition is a list of labeled candidates, that represents the ground truth
    contains the actual density map
    """
    def __init__(self, density_map, composition):
        self.density_map = density_map      # numpy 3d array
        self.composition = composition      # list of labeled candidates

