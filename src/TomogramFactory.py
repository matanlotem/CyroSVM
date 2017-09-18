from TomogramGenerator import TomogramGenerator, tomogram_loader, tomogram_generator_random
from Configuration import CONFIG

class TomogramFactory:
    def __init__(self, kind):
        self.kind = kind
        self.save = False
        self.paths = None
        self.templates = None
        self.criteria = None
        self.num_tomograms = None

    def set_templates(self, templates):
        self.templates = templates
        return self

    def set_paths(self, paths):
        self.paths = paths
        return self

    def set_save(self, value):
        self.save = value
        return self

    def set_criteria(self, value):
        self.criteria = value
        return self

    def set_num_tomograms(self, num_tomograms):
        self.num_tomograms = num_tomograms
        return self

    def build(self):
        # Assert that all the required values are set

        # build the appropriate generator
        if self.kind == TomogramGenerator.LOAD:
            assert self.paths is not None
            return tomogram_loader(self.paths)
        elif self.kind == TomogramGenerator.RANDOM:
            assert self.templates
            assert self.criteria
            assert self.num_tomograms
            return tomogram_generator_random(self.templates, self.criteria, CONFIG.DIM, self.num_tomograms)
        else:
            raise NotImplementedError('The generator %s is not implemented' % str(self.kind))
