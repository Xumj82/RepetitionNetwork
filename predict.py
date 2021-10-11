from utility import get_repnet_model, read_video, get_counts

import os

import numpy as np
import pandas as pd
import pathlib

import cv2
import glob
import random
from random import randint
import tensorflow as tf

def get_frames(path):
    frames = []
    cap = cv2.VideoCapture(path)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret is False:
            break
        frame = cv2.resize(frame,(112,112))
        frames.append(frame)
    
    cap.release()
    
    return frames

def get_countix_dataset(path):
    df = pd.read_csv(path)
    return df

def get_combined_video(path,count):

    curFrames = get_frames(path)
        
    output_len = min(len(curFrames), randint(44, 64))
            
    newFrames = []
    for i in range(1, output_len + 1):
        newFrames.append(curFrames[i * len(curFrames)//output_len  - 1])

    a = randint(0, 64 - output_len)
    b = 64 - output_len - a

    randpath = random.choice(glob.glob('data/synthvids/train*.mp4'))
    randFrames = get_frames(randpath)
    newRandFrames = []
    for i in range(1, a + b + 1):
        newRandFrames.append(randFrames[i * len(randFrames)//(a+b)  - 1])

    same = np.random.choice([0, 1], p = [0.5, 0.5])
    if same:
        finalFrames = [newFrames[0] for i in range(a)]
        finalFrames.extend( newFrames )        
        finalFrames.extend([newFrames[-1] for i in range(b)] )
    else:
        finalFrames = newRandFrames[:a]
        finalFrames.extend( newFrames )        
        finalFrames.extend( newRandFrames[a:] )
    X = []
    for img in finalFrames:
        img_tensor = tf.convert_to_tensor(img)
        img_tensor = tf.image.resize(img_tensor, [112, 112])
        img_tensor = tf.cast(img_tensor, tf.float32)
        img_tensor = img_tensor / 255.0
        X.append(img_tensor)
    X = tf.expand_dims(X, axis=0)

    y = [0 for i in range(0,a)]
    y.extend([output_len/count if 1<output_len/count<32 else 0 for i in range(0, output_len)])
    
    y.extend( [ 0 for i in range(0, b)] )
    y = tf.convert_to_tensor(y)
    y = tf.expand_dims(y, -1)
   
    save_video('test.avi',finalFrames)

    return finalFrames, y

def save_video(path,frames):
    l = len(frames)
    videoWriter = cv2.VideoWriter(path, cv2.VideoWriter_fourcc('I', '4', '2', '0'), 20, (112,112))
    i = 0
    while(i<len(frames)):
        # 展示一帧
        frame =frames[i]
        videoWriter.write(frame)
        i+=1
    cv2.destroyAllWindows()

PATH_TO_CKPT = './chk_point/'

##@title 

# FPS while recording video from webcam.
WEBCAM_FPS = 16#@param {type:"integer"}

# Time in seconds to record video on webcam. 
RECORDING_TIME_IN_SECONDS = 8. #@param {type:"number"}

# Threshold to consider periodicity in entire video.
THRESHOLD = 0.2#@param {type:"number"}

# Threshold to consider periodicity for individual frames in video.
WITHIN_PERIOD_THRESHOLD = 0.5#@param {type:"number"}

# Use this setting for better results when it is 
# known action is repeating at constant speed.
CONSTANT_SPEED = False#@param {type:"boolean"}

# Use median filtering in time to ignore noisy frames.
MEDIAN_FILTER = True#@param {type:"boolean"}

# Use this setting for better results when it is 
# known the entire video is periodic/reapeating and
# has no aperiodic frames.
FULLY_PERIODIC = False#@param {type:"boolean"}

# Plot score in visualization video.
PLOT_SCORE = False#@param {type:"boolean"}

# Visualization video's FPS.
VIZ_FPS = 30#@param {type:"integer"}

model = get_repnet_model(PATH_TO_CKPT)

model.summary()

imgs, vid_fps = read_video("./data/video.mp4")


(pred_period, pred_score, within_period,
 per_frame_counts, chosen_stride) = get_counts(
     model,
     imgs,
     strides=[1,2,3,4],
     batch_size=20,
     threshold=THRESHOLD,
     within_period_threshold=WITHIN_PERIOD_THRESHOLD,
     constant_speed=CONSTANT_SPEED,
     median_filter=MEDIAN_FILTER,
     fully_periodic=FULLY_PERIODIC)



# imgs, y = get_combined_video('./data/trainvids/train0.mp4',20)

# imgs = model.preprocess(imgs)

# imgs = tf.expand_dims(imgs, axis=0)

# raw_scores, within_period_scores, _ = model(imgs)

# t = raw_scores
# w = within_period_scores