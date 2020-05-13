import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import requests, json, time
from collections import namedtuple

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

import cv2
cap = cv2.VideoCapture('v13.mp4')
sys.path.append("..")

 
# # Object detection imports
# Here are the imports from the object d etection module.

from utils import label_map_util

from utils import visualization_utils as vis_util
import tkinter as tk
from tkinter import messagebox


def vehicleOverlap(boxes,classes,scores):
  # msg = False
  # msg2 = False
  # msg3 = False
  # magThreshold = 0.3
  # angleThreshold = 10
  for i, b in enumerate(boxes[0]):
    if classes[0][i] == 3 or classes[0][i] == 6 or classes[0][i] == 8:
      if scores[0][i] > 0.5:
        for j, c in enumerate(boxes[0]):
          if (i != j) and (classes[0][j] == 3 or classes[0][j] == 6 or classes[0][j] == 8) and scores[0][j]> 0.5:
            Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')
            ra = Rectangle(boxes[0][i][3], boxes[0][i][2], boxes[0][i][1], boxes[0][i][3])
            rb = Rectangle(boxes[0][j][3], boxes[0][j][2], boxes[0][j][1], boxes[0][j][3])
            ar = rectArea(boxes[0][i][3], boxes[0][i][1],boxes[0][i][2],boxes[0][i][3])
            col_threshold = 0.6*np.sqrt(ar)
            print(area(ra, rb))
#           area(ra, rb)
            if (area(ra,rb)<col_threshold) :
              
              print("Accident Detected!!")
              return True
  return False
              # messagebox.showwarning('Information Title','Accident has been detected @ ')
              # request = requests.post('http://httpbin.org/post', data=postData)
              
              # print(request.text)
              # msg = True
              # if msg:
              #   return True 
              #   break
  #       if msg :
  #         break
  # for i, j in enumerate(boxes[0]):
  #   ra = Rectangle(boxes[0][i][3], boxes[0][i][2], boxes[0][i][1], boxes[0][i][3])
  #   rb = Rectangle(boxes[0][j][3], boxes[0][j][2], boxes[0][j][1], boxes[0][j][3])
  #   magnitude[i] = cal_magnitude(ra, rb, boxes[i])
  #   if (magnitude[i] > magThreshold):
  #     ## It is in motion
  #     msg2 = True

  # if msg2:
  #   for i in magnitude:
  #     theta = angleOfIntersection(magnitude[i], magnitude[i+1])
  #     if theta > angleThreshold
  #       msg3 = True
  #       return msg3


def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

def area(a, b):  # returns None if rectangles don't intersect
    dx = min(a.xmax, b.xmax) - max(a.xmin, b.xmin)
    dy = min(a.ymax, b.ymax) - max(a.ymin, b.ymin)
#     print (dx, dy)
#     if (dx>=0) and (dy>=0):
    return dx*dy

def rectArea(xmax, ymax, xmin, ymin):
    x = np.abs(xmax-xmin)
    y = np.abs(ymax-ymin)
    return x*y

def cal_magnitude(i, j, u):
  return math.sqrt((u*i)**2 + (u*j)**2)

def angleOfIntersection(x, y):
  return np.arccos((x*y)/abs(x) * abs(y))

def grossSpeed(x, y, t, interval):
  interval = 5
  return (y - x)/(t*interval)

def scaledSpeed(videoHeight, carBox, grossSpeed):
  return (  ( (videoHeight - carBox)/videoHeight) + 1  ) * grossSpeed

def acceleration(scaledSpeed1, scaledSpeed2, time, interval):
  return (  (scaledSpeed2 - scaledSpeed1) / (time * interval)  )

def main_process():
  # response = {"cameraId": 42, "time": time.ctime(time.time())}
  sys.path.append("..")
  import cv2
  from utils import label_map_util

  from utils import visualization_utils as vis_util
  MODEL_NAME = 'ssd_mobilenet_v2_coco'
  MODEL_FILE = MODEL_NAME + '.tar.gz'

  PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

  PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

  NUM_CLASSES = 90


  # ## Load a (frozen) Tensorflow model into memory.
  detection_graph = tf.Graph()
  with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    print(PATH_TO_CKPT)
    with tf.compat.v1.gfile.Open(PATH_TO_CKPT, mode='rb') as fid:
      serialized_graph = fid.read()
      od_graph_def.ParseFromString(serialized_graph)
      tf.compat.v1.import_graph_def(od_graph_def, name='')

  # # ## Loading label map
  label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
  categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
  category_index = label_map_util.create_category_index(categories)

  # # Detection

  # For the sake of simplicity we will use only 2 images:
  PATH_TO_TEST_IMAGES_DIR = 'images/train'
  TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.JPG'.format(i)) for i in range(3314, 10845) ]

  # Size, in inches, of the output images.
  IMAGE_SIZE = (12, 8)

  cap = cv2.VideoCapture('crash_comp.mp4') 
  with detection_graph.as_default():
    with tf.compat.v1.Session(graph=detection_graph) as sess:
      while True:
        ret, image_np = cap.read()
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        scores = detection_graph.get_tensor_by_name('detection_scores:0')
        classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')
        (boxes, scores, classes, num_detections) = sess.run(
            [boxes, scores, classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})
        # Visualization of the results of a detection.
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=8)
        if vehicleOverlap(boxes, classes, scores):
          response = {"cameraId": 42, "time": time.ctime(time.time())}
          print(response)
          # cv2.destroyAllWindows()
          # break

        # cv2.imshow('object detection', image_np)
        cv2.imshow('object detection', image_np)

        if cv2.waitKey(25) & 0xFF == ord('q'):
          cv2.destroyAllWindows()
          response = {"cameraId": None, "time": None}
          return response
          break
  return response

if __name__ == "__main__":
  
  response = main_process()
  print(response) 
