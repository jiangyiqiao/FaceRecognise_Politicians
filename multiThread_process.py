#coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from threading import Thread
from scipy import misc
import sys
import os
import argparse
import tensorflow as tf
import numpy as np
import facenet
from align import detect_face
import random
from time import sleep

import math
import pickle
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.svm import SVC

count_rightpic=0
wrong_imagepaths=[]

input_dir='./images/policy/'
image_size=182
margin=44
gpu_memory_fraction=1.0
etect_multiple_faces=True
model_path='models/policy/embeding.pb'
classifier_filename='models/policy/svm_classifier.pkl'
batch_size=90
facenet_image_size=160


def parsePicture(picture_paths,model,class_names):
    num=0
    for i in range(0,len(picture_paths),5):
        if i+5 > len(picture_paths):return
        tasks = picture_paths[i:i+5]
        threads = []
        print("progress "+str(i)+'/'+str((len(picture_paths))))
        for url in tasks:
            print("url")
            print(url)
            num+=1
            print(num)
            t = Thread(target=embeding, args=(url,model,class_names))
            threads.append(t)
            t.start()
        for t in threads:
            t.join()


#计算向量化数据
def embeding(image_path,model,class_names):

    print(image_path)
    print(image_size)
    images ,bounding_boxes= load_and_align_data(image_path)

    # Get input and output tensors
    with tf.Graph().as_default():
        with tf.Session() as sess:
            # Load the model
            facenet.load_model(model_path)

            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            predictions=[]
            # Run forward pass to calculate embeddings
            for image in images:
                feed_dict = {images_placeholder: [np.array(image)], phase_train_placeholder: False}
                emb_datas = sess.run(embeddings, feed_dict=feed_dict)
                print("emb_datas")

                print('Testing classifier')
                prediction = model.predict(emb_datas)
                print(prediction)
                predictions.append(class_names[prediction[0]])
            
        #每张图所有预测人脸结果
    for prediction in predictions:
        print("prediction:%s"%prediction)

    nrof_faces = bounding_boxes.shape[0]  # number of faces
    print("nrof_faces:%d"%nrof_faces)
            
    #遍历每个人脸检测框
    for i,face_position in enumerate(bounding_boxes):
        face_position = face_position.astype(int)

    image_label=os.path.basename(os.path.dirname(image_path))
    if image_label in predictions:
        global count_rightpic
        count_rightpic=count_rightpic+1
        print("count_rightpic: %d"%count_rightpic)
    else:
        global wrong_imagepaths
        wrong_imagepaths.append(image_path)



def load_and_align_data(image_path):

    minsize = 20 # minimum size of face
    threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    factor = 0.709 # scale factor
    
    print('Creating networks and loading parameters')
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, None)

    img_list = []
    print(image_path)
    img = misc.imread(image_path, mode='RGB')
   
    img_size = np.asarray(img.shape)[0:2]
    bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)

    if len(bounding_boxes) < 1:
        image_paths.remove(image)
        print("can't detect face, remove ", image)
    else:
        for bounding_box in bounding_boxes:
            det = np.squeeze(bounding_box)
            bb = np.zeros(4, dtype=np.int32)
            bb[0] = np.maximum(det[0]-margin/2, 0)
            bb[1] = np.maximum(det[1]-margin/2, 0)
            bb[2] = np.minimum(det[2]+margin/2, img_size[1])
            bb[3] = np.minimum(det[3]+margin/2, img_size[0])
            cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
            aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
            prewhitened = facenet.prewhiten(aligned)
            img_list.append(prewhitened)
    return img_list,bounding_boxes



if __name__=='__main__':
    dataset = facenet.get_dataset(input_dir)
    paths, labels = facenet.get_image_paths_and_labels(dataset)      
    
    print('Number of classes: %d' % len(dataset))
    print('Number of images: %d' % len(paths))
            
    classifier_filename_exp = os.path.expanduser(classifier_filename)
    print(classifier_filename_exp)
    with open(classifier_filename_exp, 'rb') as infile:
        (model, class_names) = pickle.load(infile)
#多线程处理
    parsePicture(paths,model,class_names)
    
 #计算召回率
    print(wrong_imagepaths)
    print("count_rightpic:%d"%count_rightpic)  
    print("len(paths):%d"%len(paths))
    print("recall:")
    print(format(float(count_rightpic)/float(len(paths)),'.3f'))

