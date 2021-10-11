# RepetitionNetwork
Please run this project in vscode, existing scripts for predict/train/test for vscode debug

# Dataset
There are 2 types of dataset for using
1. Combined dataset from real videos(countix) for train and test
2. Synthetic dataset from random selected videos for test

Each of them will reture a collection of x(64 frames) and y(periodicity of each frame)

Video data can be downloaded from [here](https://drive.google.com/drive/folders/1NsZ2-Ko5eES921J1piVBs6hxW0yEDwpa?usp=sharing)

# Tensorboard
A log file will be generated after every epoch, so you can review the accuracy and loss value for every epoch, and the similiarity mitrix is also available in tensorboard
```
tensorboard --logdir logs/gradient_tape
```
An existing log file at [here](https://drive.google.com/drive/folders/17jyIB8sAVINg25L368PVJifMOxjfdUa2?usp=sharing)

