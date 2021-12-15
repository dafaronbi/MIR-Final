import numpy as np
import ds
import tensorflow as tf
from tensorflow.keras import layers, losses
import librosa
import librosa.display
import matplotlib.pyplot as plt

#load data
train_data = ds.melParamData("train","data")
test_data = ds.melParamData("test","data")
validation_data = ds.melParamData("validation","data")

print(train_data.get_mels().shape)
print(test_data.get_mels().shape)
print(validation_data.get_mels().shape)

fig, ax = plt.subplots()
img = librosa.display.specshow(train_data.get_mels()[8998], y_axis='mel', x_axis='time', ax=ax)

ax.set(title='Mel-frequency power spectrogram')
ax.label_outer()
fig.colorbar(img, ax=ax, format="%+2.f dB")
plt.show()