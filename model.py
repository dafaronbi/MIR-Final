import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import layers, losses


class Autoencoder(Model):
    """Autoencoder model with user defined latent dimensions"""

    def __init__(self,latent_dim,input_dim, output_dim):
        """
        Constructor: create instance of class
        :param latent_dim (numpy.shape): size of latent dimensions
        :param input_dim (numpy.shape): size of input dimensions
        :param output_dim (numpy.shape): size of output dimensions
        """

        #run parent constructor
        super(Autoencoder,self).__init__()

        #set dimensions of auto encoder
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.output_dim = output_dim

        #make encoder
        self.encoder = tf.keras.Sequential([
            layers.Flatten(),
            layers.Dense(latent_dim, activation='relu')
        ])

        #make decoder
        self.decoder = tf.keras.Sequential([
            layers.Dense(784, activation='sigmoid'),
            layers.Reshape(self.input_dim)
        ])

    def call(self, x):
        """
        call: create output of model of specific example
        :param x: sample to propagate
        :return: output of model
        """
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded