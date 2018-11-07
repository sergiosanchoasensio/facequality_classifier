import random
import cv2
import numpy as np
import imutils
from numpy.random import randint
import params


def horizontal_flip(image):
    return np.flip(image, 1)


def random_crop(image):
    width = image.shape[0]
    height = image.shape[1]
    crop_width = int(width * 0.33)
    crop_height = int(height * 0.33)
    x1 = np.random.randint(0, crop_width)
    x2 = np.random.randint(width - crop_width, width)
    y1 = np.random.randint(0, crop_height)
    y2 = np.random.randint(height - crop_height, height)
    return cv2.resize(image[x1:x2, y1:y2], (params.OUT_WIDTH, params.OUT_HEIGHT))


def lower_resolution(image):
    scale = random.uniform(0.3, 0.9)
    new_width = int(image.shape[1] * scale)
    new_height = int(image.shape[0] * scale)
    return cv2.resize(cv2.resize(image, (new_width, new_height)), (params.OUT_WIDTH, params.OUT_HEIGHT))


def random_patch(image):
    vertical_horizontal = np.random.randint(0, 2) < 1
    patch_scale = random.uniform(0.1, 0.5)
    if vertical_horizontal:
        width = image.shape[1]
        patch_width = int(width * patch_scale)
        patch_position = np.random.randint(0, width - patch_width)
        image[:, patch_position:patch_position + patch_width] = (127, 127, 127)
        return image
    else:
        height = image.shape[0]
        patch_height = int(height * patch_scale)
        patch_position = np.random.randint(0, height - patch_height)
        image[patch_position:patch_position + patch_height, :] = (127, 127, 127)
        return image


def blur(image):
    ks = random.randint(3, 9)
    kernel = np.ones((ks, ks), np.float32) / ks ** 2
    return cv2.filter2D(image, -1, kernel)


def motion_blur(image):
    size = random.randint(5, 25)
    angle = random.randint(1, 360)
    kernel_motion_blur = np.zeros((size, size))
    kernel_motion_blur[int((size - 1) / 2), :] = np.ones(size)
    kernel_motion_blur = kernel_motion_blur / size
    rotated_k = imutils.rotate(kernel_motion_blur, angle)
    return cv2.filter2D(image, -1, rotated_k)


def augment_tr(image):
    if randint(0, 100) < 50:
        image = horizontal_flip(image)
    if randint(0, 100) < params.PR_RANDOM_CROP:
        image = random_crop(image)
    if randint(0, 100) < params.PR_LOW_RES:
        image = lower_resolution(image)
    if randint(0, 100) < params.PR_RANDOM_PATCH:
        image = random_patch(image)
    return image