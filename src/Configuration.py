import configparser

class Configuration:
    def __init__(self):
        self.load_config('config.txt')

    def load_config(self, config_path):
        config = configparser.ConfigParser()
        config.read(config_path)
        section = config[config.sections()[0]]

        """
        Mandatory
        """
        self.DIM = section.getint('DIM')
        self.TOMOGRAM_DIMENSION = section.getint('TOMOGRAM_DIMENSION')
        self.TEMPLATE_DIMENSION = section.getint('TEMPLATE_DIMENSION')

        # For correlation search
        self.NEIGHBORHOOD_HALF_SIDE = section.getint('NEIGHBORHOOD_HALF_SIDE')

        # Candidate selector peak detection constants
        self.CORRELATION_THRESHOLD = section.getint('CORRELATION_THRESHOLD')
        self.GAUSSIAN_SIZE = section.getint('GAUSSIAN_SIZE')
        self.GAUSSIAN_STDEV = section.getint('GAUSSIAN_STDEV')

        # For Labeler
        #   This is used to determine the distance between a correlation peak and a possible Ground Truth that corresponds to it
        #   In our typical configurations, this should be
        #   2 for non noisy simulations
        #   12 for noisy 2D
        #   16 for noisy 3D
        self.DISTANCE_THRESHOLD = section.getint('DISTANCE_THRESHOLD')

        """
        Optional - but at the moment everything is mandatory
        """
        # For results metrics - if relative angle < TILT_THRESHOLD then correct tilt
        self.TILT_THRESHOLD = 15

        # For noisy tomogram
        self.NOISE_GAUSS_PARAM = section.getfloat('NOISE_GAUSS_PARAM')
        self.NOISE_LINEAR_PARAM = section.getfloat('NOISE_LINEAR_PARAM')
        self.NOISE_GAUSSIAN_SIZE = section.getint('NOISE_GAUSSIAN_SIZE')
        self.NOISE_GAUSSIAN_STDEV = section.getint('NOISE_GAUSSIAN_STDEV')

        # For chimera template generation
        self.CHIMERA_PATH = section['CHIMERA_PATH']
        self.CHIMERA_UTILS_PATH = section['CHIMERA_UTILS_PATH']

CONFIG = Configuration()
