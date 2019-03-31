#!/usr/bin/python
# -*- coding: utf-8 -*-
# pylint: disable=C0103
# pylint: disable=E1101
import matplotlib.pyplot as plt
import sys
import time
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
import os
sys.path.append("./tensorflow-face-detection")


import numpy as np
from os.path import join, exists
import keras
import tensorflow as tf

import random
import cv2
from keras import backend as K

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session()
K.set_session(session)


# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = './tensorflow-face-detection/model/frozen_inference_graph_face.pb'

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

detection_sess = tf.Session(graph=detection_graph, config=config)

def detectFace(sess,path):
    face = []
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    # Each box represents a part of the image where a particular object was detected.
    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    scores = detection_graph.get_tensor_by_name('detection_scores:0')
    classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    image = cv2.imread(path)
    image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # the array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    # Actual detection.
#     start_time = time.time()
    (boxes, scores, classes, num_detections) = sess.run(
      [boxes, scores, classes, num_detections],
      feed_dict={image_tensor: image_np_expanded})
#     elapsed_time = time.time() - start_time
    #print('inference time cost: {}'.format(elapsed_time))
    #print(boxes.shape)
    #print(scores.shape)
    #print(classes.shape,classes)
    #print(boxes.shape,scores,scores.shape)
    # Visualization of the results of a detection.
    im_height,im_width = image.shape[:2]
    for i,box in enumerate(boxes[0]):
        if scores[0][i] > 0.7:
            #print(box,box.shape)
            ymin, xmin, ymax, xmax = box
            left, right, top, bottom = int(xmin * im_width), int(xmax * im_width),int(ymin * im_height), int(ymax * im_height)
            face.append(image[top:bottom,left:right,:])
    return face 