from TemplateGenerator import TemplateGenerator, template_loader, template_generator_solid, template_generator_3d_load,\
    load_templates_3d, generate_chimera


class TemplateFactory:
    def __init__(self, kind):
        assert TemplateGenerator.contains(kind)
        self.kind = kind
        self.paths = None
        self.angular_resolution = None
        self.template_type = None

    def set_paths(self, paths):
        self.paths = paths
        return self

    def set_angular_resolution(self, resolution):
        self.angular_resolution = resolution
        return self

    def set_template_type(self, template_type):
        self.template_type = template_type
        return self

    def build(self):
        # Assert that all the required values are set
        assert self.paths is not None

        # build the appropriate generator
        if self.kind == TemplateGenerator.LOAD:
            return template_loader(self.paths)
        elif self.kind == TemplateGenerator.SOLID:
            return template_generator_solid(self.paths)
        elif self.kind == TemplateGenerator.LOAD_3D:
            return load_templates_3d(self.paths)
        elif self.kind == TemplateGenerator.CHIMERA:
            assert self.angular_resolution
            assert self.template_type
            return generate_chimera(self.paths, self.angular_resolution, self.template_type)
        else:
            raise NotImplementedError('The generator %s is not implemented' % str(self.kind))


if __name__ == '__main__':
    print(TemplateGenerator.SOLID.value)
    print(TemplateGenerator.SOLID == 'SOLID')
    print('SOLID' == TemplateGenerator.SOLID)
    print(TemplateGenerator.keys())
    print('LOAD' in TemplateGenerator.keys())
    print(TemplateGenerator.contains('LOAD'))
    print(TemplateGenerator.contains(TemplateGenerator.LOAD))
    print('HOLLOW' in TemplateGenerator.keys())
    print(TemplateGenerator.contains('HOLLOW'))
