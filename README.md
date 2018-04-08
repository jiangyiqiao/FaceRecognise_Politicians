# Politicians_FaceRecognise

## Introduction

This is a project to realise **55 Chinese politicians** face recognise. refernce the repository (**[facenet](https://github.com/davidsandberg/facenet.git)**).


## Pre-trained models
| Model name      | LFW accuracy | Training dataset | Architecture |
|-----------------|--------------|------------------|-------------|
| [20170511-185253(models/policy/embedding.pb)](https://drive.google.com/file/d/0B5MzpY9kBtDVOTVnU3NIaUdySFE) | 0.987        | CASIA-WebFace    | [Inception ResNet v1](https://github.com/davidsandberg/facenet/blob/master/src/models/inception_resnet_v1.py) |


## Dependencies
The code is tested using Tensorflow 0.12 under Ubuntu 16.04 with Python3.6 and Python3.5
* tensorflow>0.12
* sklearn
* numpy
* scipy
* pickle
* cv2
* matplotlib

I set the code:
 
    CLASS_PROBABILITY_THRESHOLD=0.2    #the probability of the predict threshold，if less than the threshold ,set the prediction="unknown"

## Dataset
1. for train set: 
    
    images/train/aligned_policy/   #person num: 55 + others, pictures num: 2409

2. for test dataset:

    images/test/others or images/test/policy

3. for train the model:

       python train_model.py

4. for test the model:

       python test_model  # I didn't set the predict threshod,the result will output the max probobility of classname. 
 

## Train your dataset 
1. put your images that haven't be aligned into the directory align/images/,like:
   
       align/images/train/policy/
           people1/
                 1.jpg
                 2.jpg
           people2/
                 1.jpg
                 2.jpg
   
2. you can change the input or output image directory 
    
       parser.add_argument('--output_dir', type=str, help='Directory with aligned face thumbnails.',default='images/train/aligned_policy/')
    

then run the code:

    python align_dataset_mtcnn.py

after run this code ,you will get the anigned_pictures,you can change the parameters to choose if you want to detect_multiple_faces,the result like:
    
    align/images/train/aligned_policy/
        people1/
               1.jpg
               2.jpg
               2_2.jpg
        people2/
               1.jpg
               1_1.jpg
               2.jpg
    
3. copy the files align/images/train/aligned_policy into images/train/ 

if you want to use my model directly,and run my project and see the result, you can

1. show the politician pictures and see the prediction or show the other people which is not the politicican one by one:
   
       python calacc_plt.py
       python calerror_plt.py
    

2. I alse provide the multi thread python code to calculate the accuracy.
    
       python multiThread_process.py
    

# Results
1. while don't set the  CLASS_PROBABILITY_THRESHOLD , can recognise profile 

![Figure_1](/result/Figure_1.png)

2. alse has some error

![Figure_1-1](/result/Figure_1-1.png)

3. can recogise sepcial part face

![Figure_1-2](/result/Figure_1-2.png)

4. can recogise sepcial part face

![Figure_1-3](/result/Figure_1-3.png)

5. as for the others

![Figure_1-5](/result/Figure_1-5.png)

6. as for the others

![Figure_1-6](/result/Figure_1-6.png)
