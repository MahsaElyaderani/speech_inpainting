import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import pickle
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Lambda
import tensorflow.keras.losses as losses
from LipNet2.predict import predict
from imageio import imread
import os

# model = load_model('/Users/kadkhodm/Desktop/Research/Mitacs/inpainting_data/results/model_2022_10_22_16_16_22',
#                   compile=False)
with open('/Users/kadkhodm/Desktop/Research/inpainting/original/datasets.cache', 'rb') as data_file:
    data = pickle.load(data_file)

## Define hyperparameters
BATCH_SIZE = 16
EPOCHS = 200
LEARNING_RATE = 0.001

MAX_ENC_SEQ_LEN = 75 #574#75
NUM_ENC_FEAT = 28 #2048
NUM_DEC_FEAT = 16
MAX_DEC_SEQ_LEN = 370 #574#75
LATENT_DIM = 16 #64  # Latent dimensionality of the encoding space.


def scale_minmax(x, min=-4):
    X_scaled = (x + (-1 * min)) / (-1 * min)
    return X_scaled


def descale_minmax(x_scaled, min=-4):

    return (x_scaled*(-1*min))+min


frames_path = sorted([os.path.join(data[1][0], x) for x in os.listdir(data[1][0])])
frames = [imread(frame_path) for frame_path in frames_path]
frames = np.array(frames)
pred = predict('/Users/kadkhodm/Desktop/Research/Mitacs/inpainting_data/results/model_2022_10_22_16_30_48',
                          frames,
                          absolute_max_string_len=32,
                          output_size=28)


"""
idx_test = 3
plt.subplot(2, 3, 1)
plt.title("decoder input")
plt.imshow(decoder_x_test[idx_test, :, :])

plt.subplot(2, 3, 2)
plt.title("decoder mask")
plt.imshow(decoder_mask_test[idx_test, :, :])

plt.subplot(2, 3, 3)
plt.title("Raw decoder output")
plt.imshow(y_pred[idx_test, :, :])

plt.subplot(2, 3, 4)
plt.title("decoder masked output")
plt.imshow(y_res[idx_test, :, :])

plt.subplot(2, 3, 5)
plt.title("Final output")
plt.imshow(y_hat[idx_test, :, :])

plt.subplot(2, 3, 6)
plt.title("Ground Truth")
plt.imshow(decoder_y_test[idx_test, :, :])

plt.show()
"""
