#import need models
import os
import tensorflow as tf
from tensorflow.keras import losses
import model
import ds
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

#load data
train_data = ds.melParamData("train","data")
test_data = ds.melParamData("test","data")
validation_data = ds.melParamData("validation","data")

#directory for finding checkpoints
checkpoint_path = "saved_models/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

#get latest model
latest = tf.train.latest_checkpoint(checkpoint_dir)

#create autoencoder model
autoencoder = model.autoencoder(64,train_data.get_mels()[:10,...,np.newaxis].shape,train_data.get_params().shape)

#load stored weights
autoencoder.load_weights(latest)

#compile model
autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())

# #print evaluation on test set
# loss, loss1,loss2 = autoencoder.evaluate(test_data.get_mels(),[test_data.get_mels(),test_data.get_params()])
# print("model loss = " + str(loss) + "\n model spectrogram loss 1 = "+ str(loss1) + "\n model synth_param loss 2 = "+ str(loss2))

#get prediction
spectogram,params = autoencoder.predict(test_data.get_mels()[[20]])

#evaluate reconstruction of 30th test file
fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
img = librosa.display.specshow(test_data.get_mels()[20], y_axis='mel', x_axis='time', ax=ax[0])

ax[0].set(title='Mel-Frequency Spectrogram Reconstruction')
ax[0].label_outer()

librosa.display.specshow(np.squeeze(spectogram), y_axis='mel', x_axis='time', ax=ax[1])
fig.colorbar(img, ax=ax, format="%+2.f dB")
plt.show()