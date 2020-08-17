import random
from typing import List, Callable
from torchvision import transforms as TF
from torchvision.transforms import functional as TFF


class Augment:

    def __init__(self, p_flip: float = 0.5, p_rot: float = 1.0, transforms: List[Callable] = None):

        if transforms is None:
            transforms = [
                TF.RandomHorizontalFlip(p=p_flip),
                TF.RandomVerticalFlip(p=p_flip),
                RandomRotation(p=p_rot, allowed_angles=[0, 90, 180, 270])
            ]

        self.transforms_composed = TF.Compose(transforms)

    def __call__(self, x):

        return self.transforms_composed(x)


class RandomRotation:

    def __init__(self, p: float = 1.0, allowed_angles=None):

        if allowed_angles is None:
            allowed_angles = [0, 90, 180, 270]

        self.p = p
        self.allowed_angles = allowed_angles

    def __call__(self, x):

        if random.random() < self.p:
            angle = random.choice(self.allowed_angles)
            x = TFF.rotate(x, angle)

        return x