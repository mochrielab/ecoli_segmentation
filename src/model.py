from __future__ import print_function
from __future__ import division

import tensorflow as tf


# model definition

# COMMENT
def inference(x, is_training, num_classes):
    print('inference dimensions:')
    print('input:', x.name, x.shape)
    print('encoder-------------------------------------------:')
    feature_maps = []
    filters = [32, 64, 128, 256, 512]
    for i, numfilters in enumerate(filters[:-1]):
        with tf.name_scope('conv'+str(i)+'_1/'):
            x = conv_block(x, numfilters, is_training)
            print(x.name, x.shape)
        with tf.name_scope('conv'+str(i)+'_2/'):
            x = conv_block(x, numfilters, is_training)
            print(x.name, x.shape)
        feature_maps.append(x)
        with tf.name_scope('max_pool'+str(i)):
            x = max_pool(x)
            print(x.name, x.shape)
    with tf.name_scope('conv'+str(len(filters)-1)+'_1/'):
        x = conv_block(x, filters[-1], is_training)
        print(x.name, x.shape)
    print('decoder:------------------------------------------')
    with tf.name_scope('conv'+str(len(filters)-1)+'_2/'):
        x = conv_block(x, filters[-1], is_training)
        print(x.name, x.shape)
    for i, numfilters in enumerate(filters[-2:0:-1]):
        with tf.name_scope('upsample'+str(i+len(filters))):
            x = upsample(x)
            print(x.name, x.shape)
            x = tf.concat(values=[x, feature_maps.pop()], axis=-1)
            print(x.name, x.shape)
        with tf.name_scope('conv'+str(i+len(filters))+'_1/'):
            x = conv_block(x, numfilters, is_training)
            print(x.name, x.shape)
        with tf.name_scope('conv'+str(i+len(filters))+'_2/'):
            x = conv_block(x, numfilters, is_training)
            print(x.name, x.shape)
    with tf.name_scope('upsample'+str(2*len(filters)-2)):
        x = upsample(x)
        print(x.name, x.shape)
        x = tf.concat(values=[x, feature_maps.pop()], axis=-1)
        print(x.name, x.shape)
    with tf.name_scope('conv'+str(2*len(filters)-2)+'_1/'):
        x = conv_block(x, filters[0], is_training)
        print(x.name, x.shape)
    with tf.name_scope('conv'+str(2*len(filters)-2)+'_2/'):
        x = conv(x, num_classes, is_training)
        print(x.name, x.shape)
#    with tf.name_scope('softmax'):
#        x = tf.nn.softmax(x, dim=-1)
    return x

def loss(logits, label):
    with tf.name_scope('loss'):
        shape = label.get_shape()
        label = tf.reshape(label, shape[0:-1])
        print(logits.shape, label.shape)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(\
            logits=logits, labels=label)
    return tf.reduce_mean(loss)

# def iou(logits, label):
#     with tf.name_scope('iou'):
#         logits = tf.argmax(tf.nn.softmax(logits), axis=-1)
#         print(logits.shape, label.shape)
#         logits=tf.reshape(logits, [-1])
#         label=tf.reshape(label, [-1])
#         inter=tf.reduce_sum(tf.multiply(logits,label))
#         union=tf.reduce_sum(tf.subtract(tf.add(logits,label),
#             tf.multiply(logits,label)))
#         return tf.div(inter,union) / label.get_shape()[0]

def train_op(loss):
    tf.summary.scalar('loss', loss)
    optimizer = tf.train.AdamOptimizer()
    global_step = tf.Variable(0, name='global_step', trainable=False)
    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op
    
def batch_norm(x, is_training):
    return tf.layers.batch_normalization(x, axis=-1,
        momentum=0.9, epsilon=1e-5,
        center=True, scale=True,
        beta_initializer=tf.zeros_initializer(),
        gamma_initializer=tf.ones_initializer(),
        moving_mean_initializer=tf.zeros_initializer(),
        moving_variance_initializer=tf.ones_initializer(),
        beta_regularizer=None, gamma_regularizer=None,
        training=is_training, trainable=True, name=None, reuse=None)

def conv(x, numfeatures,
        kernel_size=(3, 3), strides=(1, 1),
        stddev=0.02, use_bias=False):
    return tf.layers.conv2d(x, numfeatures, kernel_size,
        strides=strides,
        padding='same', data_format='channels_last',
        dilation_rate=(1, 1), activation=None,
        use_bias=use_bias,
        kernel_initializer=tf.truncated_normal_initializer(stddev=stddev),
        bias_initializer=tf.zeros_initializer(),
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        trainable=True, name=None, reuse=None)
    
def conv_block(x, numfilters, is_training):
    x = conv(x, numfilters)
    x = batch_norm(x, is_training)
    x = tf.nn.relu(x)
    return x

def max_pool(x, kernel_size=2, stride=2):
    return tf.layers.max_pooling2d(x, pool_size=[kernel_size, kernel_size],
                                   strides=[stride, stride], padding='valid')

def upsample(x, kernel_size=2):
    shape = x.get_shape()
    return tf.image.resize_nearest_neighbor(x,
        tf.stack([shape[1]*kernel_size, shape[2]*kernel_size]))
    
#def conv(x, numfilters, kernel_size=3, stride=1):
#    shape = tf.shape(x)
#    weights = tf.Variable(tf.truncated_normal(\
#        [kernel_size, kernel_size, shape[3], numfilters]))
#    return tf.nn.conv2d(x, weights, [1, stride, stride, 1], 'SAME') 
#def batch_norm(x, is_training):
#    if is_training:
#      mean, variance = tf.nn.moments(x, [0, 1, 2])
#      assign_mean = self.mean.assign(mean)
#      assign_variance = self.variance.assign(variance)
#      with tf.control_dependencies([assign_mean, assign_variance]):
#        return tf.nn.batch_norm_with_global_normalization(
#            x, mean, variance, self.beta, self.gamma,
#            self.epsilon, self.scale_after_norm)
#    else:
#      mean = self.ewma_trainer.average(self.mean)
#      variance = self.ewma_trainer.average(self.variance)
#      local_beta = tf.identity(self.beta)
#      local_gamma = tf.identity(self.gamma)
#      return tf.nn.batch_norm_with_global_normalization(
#          x, mean, variance, local_beta, local_gamma,
#          self.epsilon, self.scale_after_norm)

# batch normalization
#def batch_norm(x, phase_train):
#    with tf.variable_scope('bn'):
#        shape = tf.shape(x)
#        beta = tf.Variable(tf.constant(0.0, shape=[shape[3]]),
#                                     name='beta', trainable=True)
#        gamma = tf.Variable(tf.constant(1.0, shape=[shape[3]]),
#                                      name='gamma', trainable=True)
#        batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')
#        ema = tf.train.ExponentialMovingAverage(decay=0.99)
#
#        def mean_var_with_update():
#            ema_apply_op = ema.apply([batch_mean, batch_var])
#            with tf.control_dependencies([ema_apply_op]):
#                return tf.identity(batch_mean), tf.identity(batch_var)
#
#        mean, var = tf.cond(phase_train,
#                            mean_var_with_update,
#                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
#        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
#    return normed