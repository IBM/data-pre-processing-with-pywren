import logging
import os

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops

logger = logging.getLogger(__name__)

# Set seed to ensure image paths are shuffled the same way each time.
np.random.seed(1234)


def get_image_paths_and_labels(dataset):
    image_paths_flat = []
    labels_flat = []
    for i in range(int(len(dataset))):
        image_paths_flat += dataset[i].image_paths
        labels_flat += [i] * len(dataset[i].image_paths)
    return image_paths_flat, labels_flat


def get_dataset(input_directory):
    dataset = []

    classes = os.listdir(input_directory)
    classes.sort()
    nrof_classes = len(classes)
    for i in range(nrof_classes):
        class_name = classes[i]
        facedir = os.path.join(input_directory, class_name)
        if os.path.isdir(facedir):
            images = os.listdir(facedir)
            image_paths = [os.path.join(facedir, img) for img in images]
            dataset.append(ImageClass(class_name, image_paths))

    return dataset


def filter_dataset(dataset, min_images_per_class=10):
    filtered_dataset = []
    skipped_count = 0
    for i in range(len(dataset)):
        if len(dataset[i].image_paths) < min_images_per_class:
            logger.debug('Skipping class: {}'.format(dataset[i].name))
            skipped_count += 1
            continue
        else:
            filtered_dataset.append(dataset[i])
    logger.info('Skipped {} classes which had less than {} images.'.format(
        skipped_count, min_images_per_class))
    return filtered_dataset


def split_dataset(dataset, split_ratio=0.8):
    train_set = []
    test_set = []
    min_nrof_images = 2
    for cls in dataset:
        paths = cls.image_paths
        np.random.shuffle(paths)
        split = int(round(len(paths) * split_ratio))
        if split < min_nrof_images:
            continue  # Not enough images for test set. Skip class...
        train_set.append(ImageClass(cls.name, paths[0:split]))
        test_set.append(ImageClass(cls.name, paths[split:-1]))
    return train_set, test_set


class ImageClass():
    def __init__(self, name, image_paths):
        self.name = name
        self.image_paths = image_paths

    def __str__(self):
        return self.name + ', ' + str(len(self.image_paths)) + ' images'

    def __len__(self):
        return len(self.image_paths)
