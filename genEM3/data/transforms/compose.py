from typing import Sequence, Callable

class Compose:

    def __init__(self, transforms: Sequence[Callable]):

        self.transforms = transforms

    def __call__(self, x):

        for transform in self.transforms:
            x = transform(x)
        return x

    def __repr__(self):

        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'

        return format_string