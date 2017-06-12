from __future__ import print_function
from __future__ import division

import cv2
import os
import tensorflow as tf

# getting file names
data_path = "data/dataset1/"
filenames = os.listdir(data_path)
filenames.sort()
image_paths = [data_path + name for name in filenames if 'img_' in name]
label_paths = [data_path + name for name in filenames if 'test_' in name]

# create a partition vector
numfiles = len(image_paths)
test_set_size = int(0.2 * numfiles)
train_images =  image_paths[test_set_size:]
test_images =  image_paths[:test_set_size]
train_labels =  label_paths[test_set_size:]
test_labels =  label_paths[:test_set_size]

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

tfrecords_filename = 'data/dataset1.tfrecords'

writer = tf.python_io.TFRecordWriter(tfrecords_filename)

for img_path, label_path in zip(train_images, train_labels):
    
    img = cv2.imread(img_path, 0)
    label = cv2.imread(label_path, 0)    

    height = img.shape[0]
    width = img.shape[1]
        
    img_raw = img.tostring()
    label_raw = label.tostring()
    
    example = tf.train.Example(features=tf.train.Features(feature={
        'height': _int64_feature(height),
        'width': _int64_feature(width),
        'image_raw': _bytes_feature(img_raw),
        'mask_raw': _bytes_feature(label_raw)}))
    
    writer.write(example.SerializeToString())

writer.close()