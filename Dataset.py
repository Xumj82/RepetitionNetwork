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
        frames.append(frame)
    
    cap.release()
    
    return frames

def get_countix_dataset(path):
    df = pd.read_csv(path)
    return df

def get_combined_video(path,count):
    
    path = path.decode('utf-8')

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
    X = X

    y = [0 for i in range(0,a)]
    y.extend([output_len/count if 1<output_len/count<32 else 0 for i in range(0, output_len)])
    
    y.extend( [ 0 for i in range(0, b)] )
    y = tf.convert_to_tensor(y)
    y = tf.expand_dims(y, -1)
   
    return X, y
# contix labeled repetition videos(64 frames)
class CombinedDataset(tf.data.Dataset):
    def _generator(vidoe_path_list,countix):
        i = 0
        
        while i < len(vidoe_path_list):
            # Reading data (line, record) from the file
            X,y = get_combined_video(vidoe_path_list[i],countix[i])
            yield X,y
            i += 1

    def __new__(cls, video_path, countix_path):
        #get path collection of videos
        data_dir = video_path
        data_root = pathlib.Path(data_dir)
        all_video_path = list(data_root.glob('*'))
        all_video_path = [str(path) for path in all_video_path]

        #get count number from csv file
        countix = get_countix_dataset(countix_path)['count'].to_numpy()

        #generate video (64*112*112*3)
        dataset = tf.data.Dataset.from_generator(
            cls._generator,
            output_signature = (
                tf.TensorSpec(shape = (64, 112, 112, 3), dtype = tf.float32),
                tf.TensorSpec(shape=(64,1), dtype=tf.float64),
                ),
            args=[all_video_path,countix]
        )
        return dataset

def compress_frames(frames, output_length):
    
    new_frames = []
    for i in range(1, output_length + 1):
        new_frames.append(frames[i * len(frames)//output_length  - 1])
        
    assert(len(new_frames) == output_length)
    return new_frames

def get_rep_video(path):
    path = path.decode('utf-8')
    path +='train*.mp4'
    while True:
        path = random.choice(glob.glob(path))
        assert os.path.exists(path), "No file with this pattern exist" + path

        cap = cv2.VideoCapture(path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total > 64:
            break
        else:
            # os.remove(path)
            print('Error occur when random select from '+path)
    
    mirror = np.random.choice([0, 1], p = [0.8, 0.2])
    halfperiod = randint(2 , 31) // (mirror + 1)
    period = (mirror + 1) * halfperiod
    count = randint(max(2, 16//period), 64//(period))
    
    clipDur = randint(min(total//(64/period - count + 1), max(period, 30)), 
                        min(total//(64/period - count + 1), 60))

    repDur = count * clipDur
    noRepDur =  int((64 / (period*count) - 1) * repDur)
        
    assert(noRepDur >= 0)
    begNoRepDur = randint(0,  noRepDur)
    endNoRepDur = noRepDur - begNoRepDur
    totalDur = noRepDur + repDur
        
    startFrame = randint(0, total - (clipDur + noRepDur))
    cap.set(cv2.CAP_PROP_POS_FRAMES, startFrame)
    
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if ret is False or len(frames) == clipDur + noRepDur:
            break
        frame = cv2.resize(frame , (112, 112), interpolation = cv2.INTER_AREA)
        frames.append(frame)
    
    cap.release()
    
    
    numBegNoRepFrames = begNoRepDur*64//totalDur
    periodLength = np.zeros((64, 1))
    begNoRepFrames = compress_frames(frames[:begNoRepDur], numBegNoRepFrames)
    finalFrames = begNoRepFrames
    
    repFrames = frames[begNoRepDur : -endNoRepDur]
    repFrames.extend(repFrames[::-1])

    if len(repFrames) >= period:
        curf = numBegNoRepFrames
        for i in range(count):
            if period > 18:
                noisyPeriod = np.random.choice([max(period-1, 2), period, min(31, period + 1)])
                noisyPeriod = min(noisyPeriod, 64 - curf)
            else:
                noisyPeriod = period
            noisyFrames = compress_frames(repFrames, noisyPeriod)
            finalFrames.extend(noisyFrames)

            for p in range(noisyPeriod):
                
                try:
                    periodLength[curf] = noisyPeriod
                except: 
                    print(curf, numBegNoRepFrames, totalDur, begNoRepDur)
                assert(noisyPeriod < 32)
                curf+=1
                                            
    else:
        period = 0
        
    numEndNoRepFrames = 64 - len(finalFrames) 
    endNoRepFrames = compress_frames(frames[-endNoRepDur:], numEndNoRepFrames)
    finalFrames.extend(endNoRepFrames)
    
    # frames = randomTransform(finalFrames)
    frames=[]
    for img in finalFrames:
        img_tensor = tf.convert_to_tensor(img)
        img_tensor = tf.image.resize(img_tensor, [112, 112])
        img_tensor = tf.cast(img_tensor, tf.float32)
        img_tensor = img_tensor / 255.0
        frames.append(img_tensor)
    
    numBegNoRepFrames = begNoRepDur*64//totalDur
    if count == 1:
        numEndNoRepFrames = 64 - numBegNoRepFrames
        period = 0
        
    #assert(len(frames) == 64)
    
    #frames = F.dropout(frames, p = 0.1)
    X = tf.expand_dims(frames, axis=0)
    y = tf.convert_to_tensor(periodLength)
    return X, y
# Systhetic repetition videos(64 frames)
class SyntheticDataset(tf.data.Dataset):
    def _generator(path,sample_size):
        i = 0
        while i <sample_size:
            # Reading data (line, record) from the file
            X,y = get_rep_video(path)
            yield X,y
            i += 1

    def __new__(cls, path,sample_size):
        #get path collection of videos
        # data_dir = path
        # data_root = pathlib.Path(data_dir)
        # all_video_path = list(data_root.glob('*'))
        # all_video_path = [str(path) for path in all_video_path]
        
        #generate video (64*112*112*3)
        dataset = tf.data.Dataset.from_generator(
            cls._generator,
            output_signature = (
                tf.TensorSpec(shape = (64, 112, 112, 3), dtype = tf.float32),
                tf.TensorSpec(shape=(64, 1), dtype=tf.float64),
                ),
            args=[path,sample_size]
        )
        # dataset = tf.data.Dataset.zip((video_dataset, label_dataset))
        return dataset

