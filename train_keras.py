import tensorflow as tf
from tensorflow.keras.utils import plot_model
from Dataset import CombinedDataset,SyntheticDataset

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, losses
import numpy as np

from RepNet import ResnetPeriodEstimator

import time
import datetime
import os

from tqdm import tqdm