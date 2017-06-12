#batches

from __future__ import print_function
from __future__ import division

import matplotlib.pyplot as plt
import tensorflow as tf
import random

IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
OFFSET_HEIGHT = 32
OFFSET_WIDTH = 32


def read_and_decode(filename_queue, random_crop=True):
    
    reader = tf.TFRecordReader()

    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
      serialized_example,
      # Defaults are not specified since both keys are required.
      features={
        'height': tf.FixedLenFeature([], tf.int64),
        'width': tf.FixedLenFeature([], tf.int64),
        'image_raw': tf.FixedLenFeature([], tf.string),
        'mask_raw': tf.FixedLenFeature([], tf.string)
        })

    # Convert from a scalar string tensor (whose single string has
    # length mnist.IMAGE_PIXELS) to a uint8 tensor with shape
    # [mnist.IMAGE_PIXELS].
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    annotation = tf.decode_raw(features['mask_raw'], tf.uint8)
    
    height = tf.cast(features['height'], tf.int32)
    width = tf.cast(features['width'], tf.int32)
    
    image_shape = tf.stack([height, width, 1])
    annotation_shape = tf.stack([height, width, 1])
    
    image = tf.reshape(image, image_shape)
    annotation = tf.reshape(annotation, annotation_shape)
    
       
    # Random transformations can be put here: right before you crop images
    # to predefined size. To get more information look at the stackoverflow
    # question linked above.
    if random_crop:
        offset_height = random.randint(0, OFFSET_HEIGHT-1)
        offset_width = random.randint(0, OFFSET_WIDTH-1)
    else:
        offset_height = OFFSET_HEIGHT/2
        offset_width = OFFSET_WIDTH/2

    resized_image = tf.image.crop_to_bounding_box(image,
                                       offset_height, offset_width,
                                       target_height=IMAGE_HEIGHT,
                                       target_width=IMAGE_WIDTH)

    resized_annotation = tf.image.crop_to_bounding_box(image,
                                       offset_height, offset_width,
                                       target_height=IMAGE_HEIGHT,
                                       target_width=IMAGE_WIDTH)
    
    images, annotations = tf.train.shuffle_batch( [resized_image, resized_annotation],
                                                 batch_size=2,
                                                 capacity=30,
                                                 num_threads=2,
                                                 min_after_dequeue=10)
    
    return images, annotations

    
    
tfrecords_filename = 'data/dataset1_train.tfrecords'    
    
    
filename_queue = tf.train.string_input_producer(
    [tfrecords_filename], num_epochs=10)

# Even when reading in multiple threads, share the filename
# queue.
image, annotation = read_and_decode(filename_queue)

# The op for initializing the variables.
init_op = tf.group(tf.global_variables_initializer(),
                   tf.local_variables_initializer())

with tf.Session()  as sess:
    
    sess.run(init_op)
    
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    
    # Let's read off 3 batches just for example
    for i in range(3):
    
        img, anno = sess.run([image, annotation])
        print(img[0, :, :, :].shape)
        
        print('current batch')
        
        fig = plt.figure()
        plt.imshow(img[0,:,:,0], cmap = "gray")

        
    
    coord.request_stop()
    coord.join(threads)
