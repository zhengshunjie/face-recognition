from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
import re
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.platform import gfile


'''
Resface20 and Resface36 proposed in sphereface and applied in Additive Margin Softmax paper
Notice:
batch norm is used in line 111. to cancel batch norm, simply commend out line 111 and use line 112
'''

def prelu(x):
    with tf.variable_scope('PRelu'):   
        alphas = tf.Variable(tf.constant(0.25,dtype=tf.float32,shape=[x.get_shape()[-1]]),name='prelu_alphas')
        pos = tf.nn.relu(x)
        neg = alphas * (x - abs(x)) * 0.5
        return pos + neg

def resface_block(lower_input,output_channels,scope=None):
    with tf.variable_scope(scope):
        net = slim.conv2d(lower_input, output_channels,weights_initializer=tf.truncated_normal_initializer(stddev=0.01))
        net = slim.conv2d(net, output_channels,weights_initializer=tf.truncated_normal_initializer(stddev=0.01))
        return lower_input + net

def resface_pre(lower_input,output_channels,scope=None):
    net = slim.conv2d(lower_input, output_channels, stride=2, scope=scope)
    return net

def resface20(images, keep_probability, 
             phase_train=True, bottleneck_layer_size=512, 
             weight_decay=0.0, reuse=None):
    '''
    conv name
    conv[conv_layer]_[block_index]_[block_layer_index]
    '''
    with tf.variable_scope('Conv1'):
        net = resface_pre(images,64,scope='Conv1_pre')
        net = slim.repeat(net,1,resface_block,64,scope='Conv1')
    with tf.variable_scope('Conv2'):
        net = resface_pre(net,128,scope='Conv2_pre')
        net = slim.repeat(net,2,resface_block,128,scope='Conv2')
    with tf.variable_scope('Conv3'):
        net = resface_pre(net,256,scope='Conv3_pre')
        net = slim.repeat(net,4,resface_block,256,scope='Conv3')
    with tf.variable_scope('Conv4'):
        net = resface_pre(net,512,scope='Conv4_pre')
        net = slim.repeat(net,1,resface_block,512,scope='Conv4')

    with tf.variable_scope('Logits'):
        #pylint: disable=no-member
        #net = slim.avg_pool2d(net, net.get_shape()[1:3], padding='VALID',
        #                      scope='AvgPool')
        net = slim.flatten(net)

        net = slim.dropout(net, keep_probability, is_training=phase_train,
                           scope='Dropout')
    
    net = slim.fully_connected(net, bottleneck_layer_size, activation_fn=None, 
            scope='Bottleneck', reuse=False)            
    return net,''

def resface36(images, keep_probability, 
             phase_train=True, bottleneck_layer_size=512, 
             weight_decay=0.0, reuse=None):
    '''
    conv name
    conv[conv_layer]_[block_index]_[block_layer_index]
    '''
    with tf.variable_scope('Conv1'):
        net = resface_pre(images,64,scope='Conv1_pre')
        net = slim.repeat(net,2,resface_block,64,scope='Conv_1')
    with tf.variable_scope('Conv2'):
        net = resface_pre(net,128,scope='Conv2_pre')
        net = slim.repeat(net,4,resface_block,128,scope='Conv_2')
    with tf.variable_scope('Conv3'):
        net = resface_pre(net,256,scope='Conv3_pre')
        net = slim.repeat(net,8,resface_block,256,scope='Conv_3')
    with tf.variable_scope('Conv4'):
        net = resface_pre(net,512,scope='Conv4_pre')
        #net = resface_block(Conv4_pre,512,scope='Conv4_1')
        net = slim.repeat(net,1,resface_block,512,scope='Conv4')

    with tf.variable_scope('Logits'):
        #pylint: disable=no-member
        #net = slim.avg_pool2d(net, net.get_shape()[1:3], padding='VALID',
        #                      scope='AvgPool')
        net = slim.flatten(net)
        net = slim.dropout(net, keep_probability, is_training=phase_train,
                           scope='Dropout')
    net = slim.fully_connected(net, bottleneck_layer_size, activation_fn=None, 
            scope='Bottleneck', reuse=False)    
    return net,''

def inference(image_batch, keep_probability, 
              phase_train=True, bottleneck_layer_size=512, 
              weight_decay=0.0):
    batch_norm_params = {
        'decay': 0.995,
        'epsilon': 0.001,
        'scale':True,
        'is_training': phase_train,
        'updates_collections': None,
        'variables_collections': None
    }    
    with tf.variable_scope('Resface'):
        with slim.arg_scope([slim.conv2d, slim.fully_connected], 
                             weights_initializer=tf.contrib.layers.xavier_initializer(),
                             weights_regularizer=slim.l2_regularizer(weight_decay), 
                             activation_fn=prelu,
                             normalizer_fn=slim.batch_norm,
                             #normalizer_fn=None,
                             normalizer_params=batch_norm_params):
            with slim.arg_scope([slim.conv2d], kernel_size=3):
                return resface20(images=image_batch, 
                                keep_probability=keep_probability, 
                                phase_train=phase_train, 
                                bottleneck_layer_size=bottleneck_layer_size, 
                                reuse=None)


def load_model(model, input_map=None):
    # Check if the model is a model directory (containing a metagraph and a checkpoint file)
    #  or if it is a protobuf file with a frozen graph
    model_exp = os.path.expanduser(model)
    if (os.path.isfile(model_exp)):
        print('Model filename: %s' % model_exp)
        with gfile.FastGFile(model_exp,'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, input_map=input_map, name='')
    else:
        print('Model directory: %s' % model_exp)
        meta_file, ckpt_file = get_model_filenames(model_exp)
        
        print('Metagraph file: %s' % meta_file)
        print('Checkpoint file: %s' % ckpt_file)
      
        saver = tf.train.import_meta_graph(os.path.join(model_exp, meta_file), input_map=input_map)
        saver.restore(tf.get_default_session(), os.path.join(model_exp, ckpt_file))
    
def get_model_filenames(model_dir):
    files = os.listdir(model_dir)
    meta_files = [s for s in files if s.endswith('.meta')]
    if len(meta_files)==0:
        raise ValueError('No meta file found in the model directory (%s)' % model_dir)
    elif len(meta_files)>1:
        raise ValueError('There should not be more than one meta file in the model directory (%s)' % model_dir)
    meta_file = meta_files[0]
    ckpt = tf.train.get_checkpoint_state(model_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_file = os.path.basename(ckpt.model_checkpoint_path)
        return meta_file, ckpt_file

    meta_files = [s for s in files if '.ckpt' in s]
    max_step = -1
    for f in files:
        step_str = re.match(r'(^model-[\w\- ]+.ckpt-(\d+))', f)
        if step_str is not None and len(step_str.groups())>=2:
            step = int(step_str.groups()[1])
            if step > max_step:
                max_step = step
                ckpt_file = step_str.groups()[0]
    return meta_file, ckpt_file
