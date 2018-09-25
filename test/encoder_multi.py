import os
import cv2
import numpy as np
import tensorflow as tf

from model import load_model

facenet_model_checkpoint = os.path.dirname(__file__) + '/checkpoint_multi'

class FaceEncoder:
    def __init__(self):
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))
        with self.sess.as_default():
            load_model(facenet_model_checkpoint)
            #print 'load model is done'
            # variable_names = [v.name for v in tf.all_variables()]
            # print(variable_names)

    def generate_embedding(self, face_image):
        # Get input and output tensors
        images_placeholder = tf.get_default_graph().get_tensor_by_name("img_inputs_1:0")
        #images_placeholder2 = tf.get_default_graph().get_tensor_by_name("img_inputs:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("tower_0/resnet_v1_50_1/E_BN2/Identity:0")
        #phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        dropout_rate_placeholder = self.sess.graph.get_tensor_by_name("dropout_rate:0")
        # Run forward pass to calculate embeddings
        face_image = (face_image - 127.0) / 128.0
        feed_dict = {images_placeholder:[face_image],dropout_rate_placeholder:1}
        #face_embeddings = self.sess.run(embeddings, feed_dict=feed_dict)[0]
        face_embeddings = self.sess.run(embeddings, feed_dict=feed_dict)

        return face_embeddings
