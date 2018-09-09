import os
import cv2
import numpy as np
import tensorflow as tf

from model import load_model

facenet_model_checkpoint = os.path.dirname(__file__) + '/checkpoint'

class FaceEncoder:
    def __init__(self):
        self.sess = tf.Session()
        with self.sess.as_default():
            load_model(facenet_model_checkpoint)
            #print 'load model is done'
            # variable_names = [v.name for v in tf.all_variables()]
            # print(variable_names)

    def generate_embedding(self, face_image):
        # Get input and output tensors
        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

        # Run forward pass to calculate embeddings
        face_iamge = (face_image - 127.0) / 128.0
        feed_dict = {images_placeholder: [face_image], phase_train_placeholder: False}
        #face_embeddings = self.sess.run(embeddings, feed_dict=feed_dict)[0]
        face_embeddings = self.sess.run(embeddings, feed_dict=feed_dict)

        return face_embeddings
