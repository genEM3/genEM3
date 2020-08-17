import random
import torch
from typing import Sequence, Tuple


class RandomFlip:

    def __init__(self, p: float = 0.5, flip_plane: Tuple[int, int] = None):

        if flip_plane is None:
            flip_plane = (1, 2)

        self.p = p
        self.flip_plane = flip_plane

    def __call__(self, x):

        if random.random() < self.p:
            x = torch.flip(x, self.flip_plane)

        return x


class RandomRotation90:

    def __init__(self, p: float = 1.0, mult_90: Sequence[int] = None, rot_plane: Tuple[int, int] = None):

        if mult_90 is None:
            mult_90 = [0, 1, 2, 3]

        if rot_plane is None:
            rot_plane = (1, 2)

        self.p = p
        self.mult_90 = mult_90
        self.rot_plane = rot_plane

    def __call__(self, x):

        if random.random() < self.p:
            k = random.choice(self.mult_90)
            x = torch.rot90(x, k, self.rot_plane)

        return x