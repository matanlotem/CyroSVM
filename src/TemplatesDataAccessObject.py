from CommonDataTypes import TiltedTemplate

class HelperIterator:
    """
    implements iterator capabilities over an element that supports random access
    """
    def __init__(self, size, arr):
        self.size = size
        self.index = 0
        self.arr = arr

    def __next__(self):
        if self.index == self.size:
            raise StopIteration
        val = self.arr[self.index]
        self.index += 1
        return val

class ActionDAO:
    """
    This class behaves like a 1D array but reads the elements using the given action
    """
    def __init__(self, base, dim_size, action):
        """
        This is a DataAccessObject for a 1D array, where each element is accessed using the supplied action
        :param base: using
        :param dim_size: size of the array
        :param action a function that reads the element
        """
        self.dim_size = dim_size
        self.base = base
        self.action = action

    def __len__(self):
        return self.dim_size

    def __getitem__(self, key):
        return self.action(self.base, key)

    def __iter__(self):
        return HelperIterator(self.dim_size, self)



class BidimensionalLazyFileDAO(ActionDAO):
    """
    Data Access Object that reads from a given directory files of the format X_Y.npy
    where X and Y are integers representing template_id and tilt_id correspondingly
    Uses two 1D Lazy ActionDAOs to achieve this goal
    """
    def __init__(self, directory, dim1, dim2, separator='_', suffix='.npy'):
        """
        :param directory: base directory path (string)
        :param dim1: size of first dim
        :param dim2: size of first dim
        :param separator: string that separates the first dim index and the second dim index
        :param suffix: file suffix
        """
        ActionDAO.__init__(self, directory if directory[-1] == '\\' else (directory + '\\'), dim1, None)
        #first dim returns another LazyDAO that actually reads the file only after second []
        self.action = self.action_dao_wrapper

        # Currently part of the work around
        self.dim2 = dim2
        self.separator = separator
        self.suffix = suffix
        self.template_id = None

    # These functions are a work around the lambda problem while pickling the svm_and_templates object
    # The original lambda function:
    # lambda dir, template_id: ActionDAO(dir + str(template_id) + separator, dim2,
    #                                    lambda base, tilt_id: TiltedTemplate.fromFile(base + str(tilt_id) + suffix,
    #                                                                                  tilt_id, template_id))
    def action_dao_wrapper(self, dir, template_id):
        action_dao = BidimensionalLazyFileDAO('\\', self.dim2, 0)
        action_dao.base = dir + str(template_id) + self.separator
        action_dao.action = self.tilted_template_from_file_wrapper
        action_dao.template_id = template_id
        return action_dao

    def tilted_template_from_file_wrapper(self, base, tilt_id):
        return TiltedTemplate.fromFile(base + str(tilt_id) + self.suffix, tilt_id, self.template_id)


if __name__ == '__main__':
    path = r'C:\Users\guyrom\Documents\GitHub\CryoEmSvm\Chimera\Templates'
    arr = BidimensionalLazyFileDAO(path, 3, 84)
    b = arr[2]
    c = b[5]
    assert(c.tilt_id == 5 and c.template_id == 2 and (arr[2][5].density_map == c.density_map).all() and c.density_map.any())
    d1c = 0
    d2c = 0
    for i in enumerate(arr):
        d1c += 1
        for j in enumerate(i[1]):
            d2c += 1
            assert((j[1].density_map == arr[i[0]][j[0]].density_map).all())

    assert (len(arr) == 3 and len(arr[0]) == 84)
    assert (d1c == 3 and d2c == 3 * 84)

    print('Success!')

