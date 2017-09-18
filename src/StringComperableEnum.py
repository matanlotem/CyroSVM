from enum import Enum

class StringComperableEnum(Enum):
    # Override equality checks to accept string
    def __eq__(self, other):
        if isinstance(other, StringComperableEnum):
            return super(StringComperableEnum, self).__eq__(other)
        else:
            return self.value == other

    def __ne__(self, other):
        result = self.__eq__(other)
        if result is not NotImplemented:
            return not result
        return result

    # Shortcut to the keys
    @classmethod
    def keys(cls):
        return cls.__members__.keys()

    # Define contains to enable sort of 'in' query for strings on the values in the enum
    @classmethod
    def contains(cls, item):
        if isinstance(item, cls):
            return item in cls
        else:
            return item in cls.keys()
