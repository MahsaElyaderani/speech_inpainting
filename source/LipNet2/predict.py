import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from LipNet2.model2 import LipNet
from LipNet2.decoders import Decoder
from LipNet2.helpers import labels_to_text
from LipNet2.spell import Spell
from tensorflow.keras.models import load_model
import tensorflow.keras.losses
from tensorflow.keras.utils import get_custom_objects

np.random.seed(55)

CURRENT_PATH = '/Users/kadkhodm/Desktop/Research/inpainting/' #os.path.dirname(os.path.abspath(__file__))

FACE_PREDICTOR_PATH = os.path.join(CURRENT_PATH, 'predictors', 'shape_predictor_68_face_landmarks.dat')

PREDICT_GREEDY      = False
PREDICT_BEAM_WIDTH  = 200
PREDICT_DICTIONARY  = os.path.join(CURRENT_PATH, 'dictionaries', 'grid.txt')


def predict(model_path, video, absolute_max_string_len=32, output_size=28):
    """
    print("\nLoading data from disk...")
    video = Video(vtype='face', face_predictor_path=FACE_PREDICTOR_PATH)
    if os.path.isfile(video_path):
        video.from_video(video_path)
    else:
        video.from_frames(video_path)
    print("Data loaded.\n")

    if K.image_data_format() == 'channels_first':
        img_c, frames_n, img_w, img_h = video.data.shape
    else:
        frames_n, img_w, img_h, img_c = video.data.shape
    """

    frames_n, img_w, img_h, img_c = video.shape
    lipnet = LipNet(img_c=img_c, img_w=img_w, img_h=img_h, frames_n=frames_n,
                    absolute_max_string_len=absolute_max_string_len, output_size=output_size)

    # custom_loss = {'ctc': lambda y_true, y_pred: y_pred}
    # tensorflow.keras.losses.custom_loss = custom_loss
    # adam = Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    lipnet.partial_model = load_model(model_path, compile=False)
    # print(f"comparing weights of partial model after load {np.sum(lipnet.partial_model.layers[22].get_weights()[0])}")
    # lipnet.model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=adam) ### MAHSA: changed the loss in predict to mse to work!

    # lipnet.model =load_model(model_path, custom_objects={'ctc': lambda y_true, y_pred: y_pred})

    # lipnet.model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=adam)
    # lipnet.model.load_weights(weight_path)

    # spell = Spell(path=PREDICT_DICTIONARY)
    # decoder = Decoder(greedy=PREDICT_GREEDY, beam_width=PREDICT_BEAM_WIDTH,
    #                  postprocessors=[labels_to_text, spell.sentence])

    X_data       = np.array([video.data]).astype(np.float32) / 255
    # input_length = np.array([len(video.data)])

    y_pred         = lipnet.predict(X_data)
    # result         = decoder.decode(y_pred, input_length)[0] ### MAHSA: commented the exact results

    return y_pred #(video, result)
