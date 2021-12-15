#import needed classes
import numpy as np

class melParamData():
    """"Data set class to load MelSepctrogram with synth parameter labels"""
    def __init__(self, set_type, dir):
        """
        :param set_type (string): data set type (train, test, validation)
        :param dir (string): data directory (directory where data is stored)
        """
        #set class variables
        self.set_type = set_type
        self.dir = dir

        #load in data
        if set_type == "train":
            self.mels = np.load(dir + "/train_mel_features.npy")
            self.params = np.load(dir + "/train_patches.npy")

        if set_type == "test":
            self.mels = np.load(dir + "/test_mel_features.npy")[:1000]
            self.params = np.load(dir + "/test_patches.npy")[:1000]

        if set_type == "validation":
            self.mels = np.load(dir + "/test_mel_features.npy")[1000:]
            self.params = np.load(dir + "/test_patches.npy")[1000:]

        #set input and output size
        self.input_size = self.mels[0].shape
        self.output_size = self.params[0].shape


    def __len__(self):
        return self.mels.shape[0]

    def __getitem__(self, idx):

        #get data
        mel = self.mels[idx]
        params = self.params[idx]

        return mel,params