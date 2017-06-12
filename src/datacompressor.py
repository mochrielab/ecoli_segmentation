#Check if img read properly int8/16
from __future__ import print_function
from __future__ import division

import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf

# getting file names

def get_filenames(data_path, test_ratio = 0.2):
    filenames = os.listdir(data_path)
    filenames.sort()
    image_paths = [data_path + name for name in filenames if 'img_' in name]
    label_paths = [data_path + name for name in filenames if 'test_' in name]
    
    # create a partition vector
    numfiles = len(image_paths)
    test_set_size = int(test_ratio * numfiles)
    train_images =  image_paths[test_set_size:]
    test_images =  image_paths[:test_set_size]
    train_labels =  label_paths[test_set_size:]
    test_labels =  label_paths[:test_set_size]

    return train_images, test_images, train_labels, test_labels

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def write_tfrecords(tfrecords_filename, images, labels, max_counter = 6):
    writer = tf.python_io.TFRecordWriter(tfrecords_filename)
    
    original_images = []
    counter = 0
    for img_path, label_path in zip(images, labels):
        
        img = cv2.imread(img_path, 0)
        label = cv2.imread(label_path, 0)
    
        height = img.shape[0]
        width = img.shape[1]
        
        img_raw = img.tostring()
        label_raw = label.tostring()
        
        if counter < max_counter:
            original_images.append((img, label))
        
        example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(height),
            'width': _int64_feature(width),
            'image_raw': _bytes_feature(img_raw),
            'mask_raw': _bytes_feature(label_raw)}))
        
        writer.write(example.SerializeToString())
        counter += 1
    writer.close()
    return original_images


def read_tfrecords(tfrecords_filename, max_counter = 6):
    record_iterator = tf.python_io.tf_record_iterator(path=tfrecords_filename)
    reconstructed_images = []
    counter = 0
    for string_record in record_iterator:
        
        example = tf.train.Example()
        example.ParseFromString(string_record)
        
        height = int(example.features.feature['height']
                                     .int64_list
                                     .value[0])
        
        width = int(example.features.feature['width']
                                    .int64_list
                                    .value[0])
        
        img_string = (example.features.feature['image_raw']
                                      .bytes_list
                                      .value[0])
        
        annotation_string = (example.features.feature['mask_raw']
                                    .bytes_list
                                    .value[0])
        
        img_1d = np.fromstring(img_string, dtype=np.uint8)
        reconstructed_img = img_1d.reshape((height, width))
        
        annotation_1d = np.fromstring(annotation_string, dtype=np.uint8)
        
        # Annotations don't have depth (3rd dimension)
        reconstructed_annotation = annotation_1d.reshape((height, width))
        if counter < max_counter:
            reconstructed_images.append((reconstructed_img, reconstructed_annotation))
        counter += 1
#    plt.imshow(reconstructed_img[:,:,0], cmap = "gray")
    return reconstructed_images
    
def check(original_images, reconstructed_images):
    for original_pair, reconstructed_pair in zip(original_images, reconstructed_images):
        
        img_pair_to_compare, annotation_pair_to_compare = zip(original_pair,
                                                              reconstructed_pair)
        print(np.allclose(*img_pair_to_compare))
        print(np.allclose(*annotation_pair_to_compare))
        
if __name__ == 'main':
    data_path = "data/dataset1/"      
    train_images, test_images, train_labels, test_labels = \
        get_filenames(data_path, test_ratio = 0.2)
    
    tfrecords_filename = 'data/dataset1_train.tfrecords'
    original_images = write_tfrecords(tfrecords_filename, 
                                      train_images, train_labels, max_counter = 6)
    reconstructed_images = read_tfrecords(tfrecords_filename, max_counter = 6)
    check(original_images, reconstructed_images)
    tfrecords_filename = 'data/dataset1_test.tfrecords'
    original_images = write_tfrecords(tfrecords_filename, 
                                      test_images, test_labels, max_counter = 6)
