#import needed libraries
from tensorflow.keras import layers, losses, Model

def autoencoder(latent_dim,input_dim, output_dim):
    """
    autoencoder: Create autoencoder model
    :param latent_dim (numpy.shape): size of latent dimensions
    :param input_dim (numpy.shape): size of input dimensions
    :param output_dim (numpy.shape): size of output dimensions
    """

    #set input size
    inp = layers.Input((input_dim[-3],input_dim[-2],1))

    #convolutional layers and pooling
    encoder = layers.Conv2D(4,3,1,"same", activation='relu')(inp)
    pool = layers.MaxPool2D(2,1,"same")(encoder)
    conv = layers.Conv2D(4,3,1,"same", activation='relu')(pool)
    pool2 = layers.MaxPool2D(2, 1, "same")(conv)

    #fully connected flat layers
    encoder_flat = layers.Flatten()(pool2)

    #latent dimension
    latent_layer = layers.Dense(latent_dim)(encoder_flat)

    #decoder layers to spectrogram
    decoder = layers.Dense(input_dim[-3]*input_dim[-2])(latent_layer)
    decoder_reshaped = layers.Reshape((input_dim[-3],input_dim[-2]),name='spectrogram')(decoder)

    #decoder layers to synth parameters
    decoder_b = layers.Dense(output_dim[-1],name='synth_params')(latent_layer)

    #generate model
    return Model(inputs=inp, outputs=[decoder_reshaped, decoder_b])