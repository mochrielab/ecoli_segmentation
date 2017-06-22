#batches

from __future__ import print_function
from __future__ import division

import matplotlib.pyplot as plt
import tensorflow as tf
import random

class dataloader(object):

    def __init__(self, filename_queue, 
               batchsize=64, capacity=5000, num_threads=6,
               min_after_dequeue=1000, image_height=256, image_width=256,
               crop_height=224, crop_width=224):
        self.filename_queue = filename_queue
        self.batchsize = batchsize
        self.capcity = capacity
        self.num_threads = num_threads
        self.min_after_dequeue = min_after_dequeue
        self.crop_height = crop_height
        self.crop_width = crop_width
        self.offset_height = image_height-crop_height
        self.offset_width = image_width-crop_width

    def get_batch(self, random_crop=True):
        
        reader = tf.TFRecordReader()
    
        _, serialized_example = reader.read(self.filename_queue)
    
        features = tf.parse_single_example(
          serialized_example,
          features={
            'height': tf.FixedLenFeature([], tf.int64),
            'width': tf.FixedLenFeature([], tf.int64),
            'image_raw': tf.FixedLenFeature([], tf.string),
            'mask_raw': tf.FixedLenFeature([], tf.string)
            })
    
        image = tf.decode_raw(features['image_raw'], tf.uint8)
        annotation = tf.decode_raw(features['mask_raw'], tf.uint8)
        
        height = tf.cast(features['height'], tf.int32)
        width = tf.cast(features['width'], tf.int32)
        
        image_shape = tf.stack([height, width, 1])
        annotation_shape = tf.stack([height, width, 1])
        
        image = tf.reshape(image, image_shape)
        annotation = tf.reshape(annotation, annotation_shape)
        
        if random_crop:
            offset_height = random.randint(0, self.offset_height-1)
            offset_width = random.randint(0, self.offset_width-1)
        else:
            offset_height = self.offset_height/2
            offset_width = self.offset_width/2
    
        resized_image = tf.image.crop_to_bounding_box(image,
                                           offset_height, offset_width,
                                           target_height=self.crop_height,
                                           target_width=self.crop_width)
    
        resized_annotation = tf.image.crop_to_bounding_box(image,
                                           offset_height, offset_width,
                                           target_height=self.crop_height,
                                           target_width=self.crop_width)
        
        images, annotations = tf.train.shuffle_batch(
                [resized_image, resized_annotation],
                 batch_size=self.batchsize, capacity=self.capcity,
                 num_threads=self.num_threads,
                 min_after_dequeue=self.min_after_dequeue)
        
        return images, annotations

# test reading tf records
def test(tfrecords_filename):
    filename_queue = tf.train.string_input_producer(
        [tfrecords_filename], num_epochs=10)
    dl = dataloader(filename_queue)
    image, annotation = dl.get_batch()
    
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
    
    with tf.Session()  as sess:        
        sess.run(init_op)        
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)        
        for i in range(3):        
            img, anno = sess.run([image, annotation])
            print(img[0, :, :, :].shape)
            plt.figure()
            plt.imshow(img[0,:,:,0], cmap = "gray")
    
        coord.request_stop()
        coord.join(threads)


if __name__ == 'main':
    test('data/dataset1_train.tfrecords')
