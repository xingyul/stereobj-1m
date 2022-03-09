import numpy as np
import random
import torchvision
import cv2
from PIL import Image


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, R=None, t=None, K=None):
        for trans in self.transforms:
            img, R, t, K = \
                trans(img, R, t, K)
        return img, R, t, K

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class ToTensor(object):
    def __call__(self, img, R, t, K):
        img = np.asarray(img).astype(np.float32) / 255.
        R = np.asarray(R).astype(np.float32)
        t = np.asarray(t).astype(np.float32)
        K = np.asarray(K).astype(np.float32)
        return img, R, t, K


class Normalize(object):
    def __init__(self, mean, std, to_bgr=True):
        self.mean = mean
        self.std = std
        self.to_bgr = to_bgr

    def __call__(self, img, R, t, K):
        img -= self.mean
        img /= self.std
        img = img.astype(np.float32)
        return img, R, t, K


class ColorJitter(object):
    def __init__(
        self,
        brightness=None,
        contrast=None,
        saturation=None,
        hue=None,
    ):
        self.color_jitter = torchvision.transforms.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,
        )

    def __call__(self, image, R, t, K):
        image = np.asarray(
            self.color_jitter(
                Image.fromarray(np.ascontiguousarray(image, np.uint8))))
        return image, R, t, K


class RandomRotation(object):
    def __init__(self, degrees=360):
        self.random_rotation = torchvision.transforms.RandomRotation(
            degrees=degrees
        )

    def __call__(self, image, R, t, K):
        image = np.asarray(
            self.random_rotation(
                Image.fromarray(np.ascontiguousarray(image, np.uint8))))
        return image, R, t, K


class RandomBlur(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, R, t, K):
        if random.random() < self.prob:
            sigma = np.random.choice([3, 5, 7, 9])
            image = cv2.GaussianBlur(image, (sigma, sigma), 0)
        return image, R, t, K


def make_transforms(is_train):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    if is_train is True:
        transform = Compose([
            RandomBlur(0.2),
            ColorJitter(0.1, 0.1, 0.05, 0.05),
            ToTensor(),
            Normalize(mean=mean, std=std),
        ])
    else:
        transform = Compose([
            ToTensor(),
            Normalize(mean=mean, std=std),
        ])

    return transform
