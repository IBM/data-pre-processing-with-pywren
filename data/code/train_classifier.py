import argparse
import logging
import os
import pickle
import sys
import time

import numpy as np
import tensorflow as tf
from sklearn.svm import SVC
from tensorflow.python.platform import gfile

from input_loader import (filter_dataset, split_dataset, get_dataset,
                          get_image_paths_and_labels)

logger = logging.getLogger(__name__)


def main(input_dir, model_path, output_path, batch_size, num_threads,
         num_epochs, min_images_per_class, split_ratio):
    """
    Loads images from :param input_dir, creates embeddings using a model
    defined at :param model_path, and trains a classifier outputted to
    :param output_path, then tests the model.

    :param input_dir: Path to directory containing pre-processed images
    :param model_path: Path to protobuf graph file for facenet model
    :param output_path: Path to write output pickled classifier
    :param batch_size: Batch size to create embeddings
    :param num_threads: Number of threads to utilize for queuing
    :param num_epochs: Number of epochs for each image
    :param min_images_per_class: Minimum number of images per class
    :param split_ratio: Ratio to split train/test dataset
    """

    start_time = time.time()
    train_set, test_set = _get_test_and_train_set(
        input_dir,
        min_images_per_class=min_images_per_class,
        split_ratio=split_ratio
    )

    logger.info('Creating embeddings for training set.')
    # Run the images through the pretrained model to generate embeddings.
    emb_array, label_array, class_names = \
        run_model(train_set, model_path, batch_size, num_threads,
                  num_epochs, augment=True)
    trained_model = _train_and_save_classifier(emb_array, label_array,
                                               class_names, output_path)
    logger.info(
        'Training completed in {} seconds'.format(time.time() - start_time)
    )

    # Do the evaluation with the trained model.
    start_time = time.time()
    logger.info('Creating embeddings for test set.')
    emb_array, label_array, class_names = \
        run_model(test_set, model_path, batch_size, num_threads,
                  num_epochs=1, augment=False)
    _evaluate_classifier(emb_array, label_array, trained_model, class_names)
    logger.info(
        'Testing completed in {} seconds'.format(time.time() - start_time)
    )


def run_model(data, model_path, batch_size, num_threads, num_epochs, augment):
    """
    Load and run the specified pretrained model to generate embeddings for the
    given data.
    """

    with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:

        dataset, class_names = _load_images_and_labels(data,
                                                       batch_size=batch_size,
                                                       num_threads=num_threads,
                                                       num_epochs=num_epochs,
                                                       augment=augment)

        _load_model(model_filepath=model_path)

        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        sess.run(init_op)

        images_placeholder = \
            tf.get_default_graph().get_tensor_by_name("input:0")
        embedding_layer = \
            tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = \
            tf.get_default_graph().get_tensor_by_name("phase_train:0")

        emb_array, label_array = _create_embeddings(
            embedding_layer, dataset, images_placeholder,
            phase_train_placeholder, sess
        )

        logger.info('Created {} embeddings'.format(len(emb_array)))
        return emb_array, label_array, class_names


def _get_test_and_train_set(input_dir, min_images_per_class, split_ratio=0.7):
    """
    Load train and test dataset. Classes with < :param min_num_images_per_label
    will be filtered out.
    :param input_dir:
    :param min_num_images_per_label:
    :param split_ratio:
    """
    dataset = get_dataset(input_dir)
    dataset = filter_dataset(dataset,
                             min_images_per_class=min_images_per_class)
    train_set, test_set = split_dataset(dataset, split_ratio=split_ratio)

    return train_set, test_set


def _load_images_and_labels(dataset, batch_size, num_threads, num_epochs,
                            augment=False):
    """
    Create the input pipeline for the dataset to be used in generating the
    embeddings. If :param augment is True, then small image augmentations will
    be performed on all the samples in the dataset.
    """
    class_names = [cls.name for cls in dataset]
    image_paths, labels = get_image_paths_and_labels(dataset)

    data = tf.data.Dataset.from_tensor_slices((image_paths, labels)) \
        .shuffle(len(image_paths)) \
        .repeat(num_epochs) \
        .map(_preprocess_function, num_parallel_calls=num_threads)

    if augment:
        data = data.map(_augment_function, num_parallel_calls=num_threads)

    data = data.batch(batch_size).prefetch(1)
    return data, class_names


def _preprocess_function(image_path, label):
    """
    Parse, resize, and standardize the given image.
    """
    image_size = 160
    file_contents = tf.read_file(image_path)
    image = tf.image.decode_jpeg(file_contents, channels=3)
    image = tf.random_crop(image, size=[image_size, image_size, 3])
    image.set_shape((image_size, image_size, 3))
    image = tf.image.per_image_standardization(image)
    return image, label


def _augment_function(image, label):
    """
    Perform random augmentations on the given image to boost the dataset.
    """
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.3)
    image = tf.image.random_contrast(image, lower=0.2, upper=1.8)
    return image, label


def _load_model(model_filepath):
    """
    Load frozen protobuf graph
    :param model_filepath: Path to protobuf graph
    :type model_filepath: str
    """
    model_exp = os.path.expanduser(model_filepath)
    if os.path.isfile(model_exp):
        logging.info('Model filename: %s' % model_exp)
        with gfile.FastGFile(model_exp, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')
    else:
        logger.error('Missing model file. Exiting')
        sys.exit(-1)


def _create_embeddings(embedding_layer, dataset, images_placeholder,
                       phase_train_placeholder, sess):
    """
    Uses model to generate embeddings from :param images.
    :param embedding_layer:
    :param dataset:
    :param images_placeholder:
    :param phase_train_placeholder:
    :param sess:
    :return: (tuple): image embeddings and labels
    """
    emb_array = None
    label_array = None
    try:
        i = 0
        iterator = dataset.make_one_shot_iterator()
        batch = iterator.get_next()
        while True:
            batch_images, batch_labels = sess.run(batch)
            logger.info('Processing iteration {} batch of size: {}'
                        .format(i, len(batch_labels)))
            print(batch_images)
            emb = sess.run(
                embedding_layer,
                feed_dict={images_placeholder: batch_images,
                           phase_train_placeholder: False}
            )

            emb_array = np.concatenate([emb_array, emb]) \
                if emb_array is not None else emb
            label_array = np.concatenate([label_array, batch_labels]) \
                if label_array is not None else batch_labels
            i += 1

    except tf.errors.OutOfRangeError:
        pass

    return emb_array, label_array


def _train_and_save_classifier(emb_array, label_array, class_names,
                               output_path):
    """
    Train the classifier using support vector classification and save
    the output to a pickle file.
    """
    logger.info('Training Classifier')
    model = SVC(kernel='linear', probability=True, verbose=False)
    # Fit the model according to the given training data.
    model.fit(emb_array, label_array)

    with open(output_path, 'wb') as outfile:
        pickle.dump((model, class_names), outfile)
    logging.info('Saved classifier model to file "%s"' % output_path)
    return model


def _evaluate_classifier(emb_array, label_array, model, class_names):
    """
    Evaluate how the trained model performed with the given embeddings and
    labels.
    """
    logger.info('Evaluating classifier on {} images'.format(len(emb_array)))

    predictions = model.predict_proba(emb_array, )
    best_class_indices = np.argmax(predictions, axis=1)
    best_class_probabilities = predictions[
        np.arange(len(best_class_indices)), best_class_indices
    ]

    for i in range(len(best_class_indices)):
        print('%4d  Prediction: %s, Confidence: %.3f, Actual: %s' % (
            i, class_names[best_class_indices[i]],
            best_class_probabilities[i], class_names[label_array[i]])
        )

    accuracy = np.mean(np.equal(best_class_indices, label_array))
    print('Accuracy: %.3f' % accuracy)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--model-path', type=str, action='store',
                        dest='model_path',
                        help='Path to model protobuf graph')
    parser.add_argument('--input-dir', type=str, action='store',
                        dest='input_dir',
                        help='Input path of data to train on')
    parser.add_argument('--output-path', type=str, action='store',
                        dest='output_path',
                        help='Path to output trained classifier model',
                        default='./output-classifier.pkl')
    parser.add_argument('--batch-size', type=int, action='store',
                        dest='batch_size', default=128,
                        help='Batch size to create embeddings')
    parser.add_argument('--num-threads', type=int, action='store',
                        dest='num_threads', default=16,
                        help='Number of threads to utilize for preprocessing.')
    parser.add_argument('--num-epochs', type=int, action='store',
                        dest='num_epochs', default=3,
                        help='Number of epochs for each image.')
    parser.add_argument('--split-ratio', type=float, action='store',
                        dest='split_ratio', default=0.7,
                        help='Ratio to split train/test dataset')
    parser.add_argument('--min-num-images-per-class', type=int, action='store',
                        default=10, dest='min_images_per_class',
                        help='Minimum number of images per class')

    args = parser.parse_args()

    main(input_dir=args.input_dir, model_path=args.model_path,
         output_path=args.output_path, batch_size=args.batch_size,
         num_threads=args.num_threads, num_epochs=args.num_epochs,
         min_images_per_class=args.min_images_per_class,
         split_ratio=args.split_ratio)
