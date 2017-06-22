from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import dataloader as dl
import model as unet
import tensorflow as tf
import os
import matplotlib.pyplot as plt

path = '/home/yz/tfseg/'
#def main():

with tf.Graph().as_default():
    with tf.name_scope('data_pipline'):
        train_filename_queue = tf.train.string_input_producer(\
                        [path+'data/dataset1_train.tfrecords',
                        # path+'data/dataset2_train.tfrecords',
                        path+'data/dataset3_train.tfrecords',
                        # path+'data/dataset4_train.tfrecords',
                        ], num_epochs=100000000)
        val_filename_queue = tf.train.string_input_producer(\
                        [path+'data/dataset1_test.tfrecords',
                        # path+'data/dataset2_test.tfrecords',
                        path+'data/dataset3_test.tfrecords',
                        # path+'data/dataset4_test.tfrecords',
                        ], num_epochs=1000000000)
        train_data_sets = dl.dataloader(train_filename_queue, batchsize=32)
        # val_data_sets = dl.dataloader(val_filename_queue, batchsize=8)
        image, label = train_data_sets.get_batch()
    logits = unet.inference(image, True, num_classes=3)
    loss = unet.loss(logits, label)
    # iou = unet.iou(logits, label)
    train_op = unet.train_op(loss)
    loss_sumary = tf.summary.scalar('loss', loss)
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
    saver = tf.train.Saver()
#    print('trainable variables:---------------------------------------')
#    for var in tf.trainable_variables():
#        print(var.name, var.shape)
    
    with tf.Session() as sess:
        summary_writer = tf.summary.FileWriter(path+'logs', sess.graph)
        sess.run(init_op)
        checkpoint_file = os.path.join(path, 'logs', 'model.ckpt')
        saver.save(sess, checkpoint_file, global_step=0)
        
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
#        for i in range(3):        
#            img, anno = sess.run([image, label])
#            print(img[0, :, :, :].shape)
#            plt.figure()
#            plt.imshow(img[0,:,:,0], cmap = "gray")
        for i in range(10000):
            loss_value, _, loss_sums = \
                sess.run([loss, train_op, loss_sumary])
            summary_writer.add_summary(loss_sums, i)
            print(loss_value)
        coord.request_stop()
        coord.join(threads)



