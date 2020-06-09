#!/usr/bin/env python3

import argparse  # Command line parsing
import configparser  # Saving the models parameters
import datetime  # Chronometer
import os  # Files management
import tensorflow as tf
import numpy as np
import math

from tqdm import tqdm  # Progress bar
from tensorflow.python import debug as tf_debug

from process_data import TextData
#from chatbot.model import Model

class Args():
    def __init__(self):
        self.autoEncode = False,
        self.batchSize = 256
        self.corpus = 'cornell'
        self.datasetTag = ''
        self.ratioDataset = 1.0
        self.maxLength = 10
        self.filterVocab = 1
        self.skipLines = False
        self.vocabularySize = 40000
        self.hiddenSize = 512
        self.numLayers = 2
        self.softmaxSamples = 0
        self.initEmbeddings = False
        self.embeddingSize = 64
        self.embeddingSource = 'GoogleNews-vectors-negative300.bin'
        self.numEpochs = 30
        self.saveEvery = 2000
        self.learningRate = 0.002
        self.dropout = 0.
        self.rootDir = None
        self.playDataset = True

class Chatbot:
    def __init__(self):
        """
        """
        # Model/dataset parameters
        self.args = Args()

        # Task specific object
        self.textData = None  # Dataset
        self.model = None  # Sequence to sequence model

        # Tensorflow utilities for convenience saving/logging
        self.writer = None
        self.saver = None
        self.modelDir = ''  # Where the model is saved
        self.globStep = 0  # Represent the number of iteration for the current model

        # TensorFlow main session (we keep track for the daemon)
        self.sess = None

        # Filename and directories constants
        self.MODEL_DIR_BASE = 'save' + os.sep + 'model'
        self.MODEL_NAME_BASE = 'model'
        self.MODEL_EXT = '.ckpt'
        self.CONFIG_FILENAME = 'params.ini'
        self.CONFIG_VERSION = '0.5'
        self.TEST_IN_NAME = 'data' + os.sep + 'test' + os.sep + 'samples.txt'
        self.TEST_OUT_SUFFIX = '_predictions.txt'
        self.SENTENCES_PREFIX = ['Q: ', 'A: ']


    def main(self, args=None):
        print('TensorFlow detected: v{}'.format(tf.__version__))

        self.args.rootDir = os.getcwd()  # Use the current working directory

        #tf.logging.set_verbosity(tf.logging.INFO) # DEBUG, INFO, WARN (default), ERROR, or FATAL
        self.textData = TextData(self.args)


if __name__ == "__main__":
    chatbot = Chatbot()
    chatbot.main()