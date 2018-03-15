from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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
def main(args):
    dataset = facenet.get_dataset(args.input_dir)

    paths, labels = facenet.get_image_paths_and_labels(dataset)          
    print('Number of classes: %d' % len(dataset))
    print('Number of images: %d' % len(paths))
            
    classifier_filename_exp = os.path.expanduser(args.classifier_filename)
    with open(classifier_filename_exp, 'rb') as infile:
        (model, class_names) = pickle.load(infile)
    
    wrong_imagepaths=[]
    for cls in dataset:
        for image_path in cls.image_paths:
            print(image_path)
            img = mpimg.imread(image_path)
            images ,bounding_boxes= load_and_align_data(image_path, args.image_size, args.margin, args.gpu_memory_fraction)

            # Get input and output tensors
            with tf.Graph().as_default():
                with tf.Session() as sess:
                    # Load the model
                    facenet.load_model(args.model)

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
                        #print(emb_datas)
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
                print("face_position：")
                print(face_position)
                
                # cv2.putText在图片上添加水印
                cv2.rectangle(img, (face_position[0],face_position[1]),(face_position[2], face_position[3]),(0, 255, 0), 1)

                cv2.putText(img, predictions[i], (face_position[0] + 5, face_position[1] + 10),cv2.FONT_HERSHEY_COMPLEX_SMALL,0.8, (255, 0 ,0),thickness = 2, lineType = 1)

            # show result
            img = cv2.resize(img, (640, 480), interpolation=cv2.INTER_CUBIC)
            plt.title(image_path)
            plt.imshow(img)
            plt.pause(2)
            #计算召回率
            image_label=os.path.basename(os.path.dirname(image_path))
            print("image_label:%s"%image_label)
            if image_label in predictions:
                global count_rightpic
                count_rightpic=count_rightpic+1
                print(count_rightpic)
            else:
                wrong_imagepaths.append(image_path)

    print(wrong_imagepaths)
    print("count_rightpic:%d"%count_rightpic)  
    print("len(paths):%d"%len(paths))
    print("recall:")
    print(format(float(count_rightpic)/float(len(paths)),'.3f'))


def load_and_align_data(image_path, image_size, margin, gpu_memory_fraction):

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
    img = misc.imread(os.path.expanduser(image_path), mode='RGB')
    img_size = np.asarray(img.shape)[0:2]
    bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)

    if len(bounding_boxes) < 1:
        #image_paths.remove(image)
        print("can't detect face, remove ", image_path)
    else:
        for bounding_box in bounding_boxes:
            print("bounding_box")
            print(bounding_box)
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



def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_dir', type=str, help='Directory with unaligned images.',default='images/test')
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=182)
    parser.add_argument('--margin', type=int,
        help='Margin for the crop around the bounding box (height, width) in pixels.', default=44)
    parser.add_argument('--random_order',
        help='Shuffles the order of images to enable alignment using multiple processes.', action='store_true')
    parser.add_argument('--gpu_memory_fraction', type=float,
        help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)
    parser.add_argument('--detect_multiple_faces', type=bool,
                        help='Detect and align multiple faces per image.', default=True)
    parser.add_argument('--model', type=str,
        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file',default='models/policy/embeding.pb')
    parser.add_argument('--classifier_filename',
        help='Classifier model file name as a pickle (.pkl) file. ' +
        'For training this is the output and for classification this is an input.',default='models/policy/svm_classifier.pkl')
    parser.add_argument('--batch_size', type=int,
        help='Number of images to process in a batch.', default=90)
    parser.add_argument('--facenet_image_size', type=int,
        help='Image size (height, width) in pixels.', default=160)
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
