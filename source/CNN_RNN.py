import os
# from tensorflow.python.keras import backend as K
import numpy as np
import tensorflow as tf
import tensorflow.keras
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Conv1D, Conv2D, Lambda, ConvLSTM2D  # , ConvLSTM1D
from tensorflow.keras.layers import Input, BatchNormalization, concatenate, Dense, LSTM, Bidirectional
from tensorflow.keras.layers import add, MultiHeadAttention, Reshape, GRU, RepeatVector, TimeDistributed
from tensorflow.keras.layers import Flatten, ZeroPadding2D, Activation, MaxPooling2D, SpatialDropout2D
from tensorflow.keras.layers import Conv3D, ZeroPadding3D
from tensorflow.keras.layers import MaxPooling3D
from tensorflow.keras.layers import Dense, Activation, SpatialDropout3D, Flatten, ReLU, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import mse, mae
import datetime
import pickle
from matplotlib import pyplot as plt
import librosa
from Speech_VGG import speechVGG
from CNN_RNN_Dataset import Dataset
from tools_audio import inv_melspectrogram
from pystoi import stoi
from pesq import pesq
import torch
from torch_pesq import PesqLoss
from joblib import Parallel, delayed
from jiwer import wer
from sample_wght_callback import SampleWeights
import pmsqe
import wandb
from wandb.keras import WandbCallback, WandbMetricsLogger, WandbModelCheckpoint, WandbEvalCallback
from tensorflow.keras.callbacks import LambdaCallback

# from wer_callback import CallbackEval
run = wandb.init(
    # set the wandb project where this run will be logged
    project="AV-SI",

    # track hyperparameters and run metadata
    config={
        "learning_rate": 1e-3,
        "architecture": "LSTM_ENC_DEC",
        "dataset": "GRID_AUG",
        "epochs": 20,
        "data_loader": True,
    }
)
config = run.config

MAX_ENC_SEQ_LEN = 149
NUM_ENC_FEAT = 68 * 2
NUM_DEC_FEAT = 64
MAX_DEC_SEQ_LEN = 149
ALGN_LEN = 42
EPOCHS = 200

characters = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]
# Mapping characters to integers
char_to_num = tensorflow.keras.layers.StringLookup(vocabulary=characters, oov_token="")
num_to_char = tensorflow.keras.layers.StringLookup(vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True)


class Seq2seq():

    def __init__(self, multi_tasking_flag, lipnet_model_path, multimodal_flag,
                 attention_flag, seq2seq_flag, lipnet_flag, landmark_flag,
                 cnn_flag, vgg_flag, mtrc_flag):

        self.model = None
        self.CURRENT_PATH = '/Users/kadkhodm/Desktop/Research/inpainting/'
        self.AUDIO_RESULTS_PATH = os.path.join(self.CURRENT_PATH, 'audio_results')
        self.SAVED_MODEL_PATH = os.path.join(self.CURRENT_PATH, 'saved_models')
        self.MODEL_LOGS_PATH = os.path.join(self.CURRENT_PATH, 'logs')
        self.SPEC_RESULTS_PATH = os.path.join(self.CURRENT_PATH, 'spec_results')
        self.run_name = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        self.multimodal_flag = multimodal_flag
        self.attention_flag = attention_flag
        self.seq2seq_flag = seq2seq_flag
        self.lipnet_model_path = lipnet_model_path
        self.lipnet_flag = lipnet_flag
        self.landmark_flag = landmark_flag
        self.multi_tasking_flag = multi_tasking_flag
        self.cnn_flag = cnn_flag
        self.vgg_flag = vgg_flag
        self.mtrc_flag = mtrc_flag
        self.MISS_WGHT = 1
        self.NON_MISS_WGHT = 1
        self.BATCH_SIZE = 32
        self.LEARNING_RATE = 1e-3
        self.LATENT_DIM = 256
        self.sample_rate = 8000

        if self.multimodal_flag and self.attention_flag and self.lipnet_flag and self.vgg_flag:
            self.model_name = "multimodal_attn_lipNet_vgg"
        elif self.multimodal_flag and self.seq2seq_flag and not self.attention_flag \
                and self.lipnet_flag and self.vgg_flag:
            self.model_name = "multimodal_seq2seq_lipNet_vgg"
        elif self.multimodal_flag and not self.seq2seq_flag and not self.attention_flag \
                and self.lipnet_flag and self.vgg_flag:
            self.model_name = "multimodal_cat_lipNet_vgg"
        elif self.multimodal_flag and not self.seq2seq_flag and not self.attention_flag \
                and not self.lipnet_flag and self.vgg_flag and not self.multi_tasking_flag:
            self.model_name = "AV_SI_landmarks_vgg"
        elif self.multimodal_flag and self.attention_flag and not self.lipnet_flag and self.vgg_flag:
            self.model_name = "AV_attn_landmarks_vgg"
        elif self.multimodal_flag and self.seq2seq_flag and not self.attention_flag \
                and not self.lipnet_flag and self.vgg_flag:
            self.model_name = "AV_S2S_landmarks_vgg"
        elif not self.multimodal_flag and not self.cnn_flag and self.vgg_flag:
            self.model_name = "A_SI_vgg"
        elif not self.multimodal_flag and self.cnn_flag and self.vgg_flag:
            self.model_name = "A_SI_cnn_vgg"
        elif self.multimodal_flag and self.attention_flag and self.lipnet_flag and not self.vgg_flag:
            self.model_name = "AV_attn_lipNet_mse"
        elif self.multimodal_flag and self.seq2seq_flag and not self.attention_flag \
                and self.lipnet_flag and not self.vgg_flag:
            self.model_name = "AV_S2S_lipNet_mse"
        elif self.multimodal_flag and not self.seq2seq_flag and not self.attention_flag \
                and self.lipnet_flag and not self.vgg_flag:
            self.model_name = "AV_SI_lipNet_mse"
        elif self.multimodal_flag and not self.seq2seq_flag and not self.attention_flag \
                and not self.lipnet_flag and not self.vgg_flag and not self.multi_tasking_flag:
            self.model_name = "AV_SI_landmarks_mse"
        elif self.multimodal_flag and self.seq2seq_flag and not self.attention_flag \
                and not self.lipnet_flag and not self.vgg_flag and not self.multi_tasking_flag:
            self.model_name = "AV_S2S_landmarks_mse"
        elif self.multimodal_flag and self.attention_flag and not self.lipnet_flag and not self.vgg_flag:
            self.model_name = "AV_attn_landmarks_mse"
        elif self.multimodal_flag and self.seq2seq_flag and not self.attention_flag \
                and not self.lipnet_flag and not self.vgg_flag and self.multi_tasking_flag:
            self.model_name = "AV_MTL_S2S_landmarks_mse"
        elif self.multimodal_flag and not self.seq2seq_flag and not self.attention_flag \
                and not self.lipnet_flag and not self.vgg_flag and self.multi_tasking_flag:
            self.model_name = "AV_MTL_SI_landmarks_mse"
        elif not self.multimodal_flag and not self.cnn_flag and not self.vgg_flag:
            self.model_name = "A_SI_mse"
        elif not self.multimodal_flag and self.cnn_flag and not self.vgg_flag:
            self.model_name = "A_SI_cnn_mse"
        self.accuracy = {}
        self.pred = {}
        self.align = {}
        self.y_res = {}
        # self.y_hat = {}
        self.data_mask = {}
        self.data_mask_not = {}
        self.build_network()
        # Building the train dataset during the class initialization
        self.ds = Dataset(self.lipnet_model_path, self.multimodal_flag,
                          self.lipnet_flag, self.landmark_flag, self.multi_tasking_flag)
        self.dataset = self.ds.dataset
        self.data = self.ds.data
        # self.decoder_max = 1
        self.df = {"train": pd.read_csv("train_data.csv"),
                   "val": pd.read_csv("val_data.csv"),
                   "test": pd.read_csv("test_data.csv")}
        if self.multi_tasking_flag:
            self.loss_func = {'align': self.ctc_loss, 'decoder_output': self.l2_loss}
            self.loss_weights = {'align': 0.001, 'decoder_output': 1}
        else:
            self.loss_func = {'decoder_output': self.l2_loss}
            self.loss_weights = {'decoder_output': 1}
        # self.data_mask['train'] = [np.where(self.data['train']["decoder_mask"][x][:, 0] == 0,
        #                                     self.MISS_WGHT, self.NON_MISS_WGHT)
        #                            for x in range(0, len(self.data['train']["decoder_mask"]))]
        # self.data_mask['val'] = [np.where(self.data['val']["decoder_mask"][x][:, 0] == 0,
        #                                   self.MISS_WGHT, self.NON_MISS_WGHT)
        #                          for x in range(0, len(self.data['val']["decoder_mask"]))]
        # self.data_mask['test'] = [np.where(self.data['test']["decoder_mask"][x][:, 0] == 0,
        #                                    self.MISS_WGHT, self.NON_MISS_WGHT)
        #                           for x in range(0, len(self.data['test']["decoder_mask"]))]

        # if self.vgg_flag and self.multi_tasking_flag:
        #     # self.sample_weight = np.array(self.data_mask['train'])
        #     self.loss_func = {'align': self.ctc_loss, 'decoder_output': self.l2_loss, 'decoder_output': self.svgg_loss}
        #     self.loss_weights = {'align': 0.001, 'decoder_output': 1, 'decoder_output': 0.001}
        #     # metrics=[self.masked_relative_err])#[self.masked_acc, self.masked_loss])
        #
        # elif self.vgg_flag and not self.multi_tasking_flag:
        #     # self.sample_weight = np.array(self.data_mask['train'])
        #     self.loss_func = {'decoder_output': self.svgg_loss, 'decoder_output': self.l2_loss}
        #     self.loss_weights = {'decoder_output': 0.01, 'decoder_output': 1}
        #
        # elif not self.vgg_flag and self.multi_tasking_flag and not self.mtrc_flag:
        #     # self.sample_weight = np.array(self.data_mask['train'])
        #     self.loss_func = {'align': self.ctc_loss, 'decoder_output': self.l1_loss}
        #     self.loss_weights = {'align': 0.001, 'decoder_output': 1}
        #
        # elif self.mtrc_flag and not self.multi_tasking_flag:
        #     print("metric flag and not multi-tasking")
        #     # self.sample_weight = np.array(self.data_mask['train'])
        #     self.loss_func = {'decoder_output': self.mtrc_loss}
        #     self.loss_weights = {'decoder_output': 1}
        #
        # elif self.mtrc_flag and self.multi_tasking_flag and not self.vgg_flag:
        #     # self.sample_weight = np.array(self.data_mask['train'])
        #     self.loss_func = {'align': self.ctc_loss, 'decoder_output': self.mtrc_loss}
        #     self.loss_weights = {'align': 0.001, 'decoder_output': 1}
        # else:
        #     # self.sample_weight = None  # np.array(self.data_mask['train'])
        #     self.loss_func = {'decoder_output': self.l2_loss}
        #     self.loss_weights = {'decoder_output': 1}
        #     metrics=[self.masked_relative_err])#[self.masked_acc, self.masked_loss])
        # if self.multimodal_flag:
        #     self.x_train = [np.array(self.data['train']["encoder_input"]),
        #                     np.array(self.data['train']["decoder_input"])]
        #     self.x_val = [np.array(self.data['val']["encoder_input"]),
        #                   np.array(self.data['val']["decoder_input"])]
        #
        # else:
        #     self.x_train = np.array(self.data['train']["decoder_input"])
        #     self.x_val = np.array(self.data['val']["decoder_input"])
        #
        # if self.multi_tasking_flag:
        #     self.y_train = [np.array(self.data['train']["align"]),
        #                     np.array(self.data['train']["decoder_target"])]
        #     self.y_val = [np.array(self.data['val']["align"]),
        #                   np.array(self.data['val']["decoder_target"])]
        # else:
        #     self.y_train = np.array(self.data['train']["decoder_target"])
        #     self.y_val = np.array(self.data['val']["decoder_target"])

        # self.perceptual_loss = self.speechVGG_model()
        # self.pesq_loss_torch = PesqLoss(0.5, sample_rate=self.sample_rate)#.float()
        # pmsqe.init_constants(Fs=8000,
        #                      Pow_factor=pmsqe.perceptual_constants.Pow_correc_factor_SqHann,
        #                      apply_SLL_equalization=True,
        #                      apply_bark_equalization=True,
        #                      apply_on_degraded=True,
        #                      apply_degraded_gain_correction=True)

    # def svgg_loss(self, y_true, y_pred):
    #
    #     exp_y_true = tf.expand_dims(y_true, axis=-1)
    #     exp_y_pred = tf.expand_dims(y_pred, axis=-1)
    #     y_true_feat = self.perceptual_loss(exp_y_true)
    #     y_pred_feat = self.perceptual_loss(exp_y_pred)
    #     percp_loss = tf.reduce_mean(mae(y_true_feat, y_pred_feat))
    #
    #     return percp_loss

    # def tv_loss(self, y_true, y_pred):
    #
    #     # exp_y_true = tf.expand_dims(y_true, axis=-1)
    #     # exp_y_pred = tf.expand_dims(y_pred, axis=-1)
    #     # tv_pred_loss = tf.image.total_variation(exp_y_pred)
    #     # tv_true_loss = tf.image.total_variation(exp_y_true)
    #     # tv_loss = tf.reduce_mean(mse(tv_true_loss, tv_pred_loss))
    #
    #     exp_y_pred = tf.expand_dims(y_pred, axis=-1)
    #     tv_loss = tf.reduce_sum(tf.image.total_variation(exp_y_pred))
    #
    #     return tv_loss

    def ctc_loss(self, y_true, y_pred):

        # label = tf.where(tf.not_equal(y_true, tf.constant(-1, y_true.dtype)))
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

        loss = tf.keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)

        return loss

    def l2_loss(self, y_true, y_pred):
        l2_loss = tf.reduce_mean((y_true - y_pred) ** 2)

        return l2_loss

    def l1_loss(self, y_true, y_pred):
        l1_loss = tf.reduce_mean(tf.abs(y_true - y_pred))
        # speech_loss = tf.reduce_mean(2 * tf.abs(y_true ** 2 - y_pred * y_true), axis=-1)
        return l1_loss

    # def mtrc_loss(self, y_true, y_pred):
    #     y_true_sq = tf.squeeze(y_true)
    #     y_pred_sq = tf.squeeze(y_pred)
    #     print('y_true shape: ', y_true)
    #     print('y_pred shape: ', y_pred)
    #     # padding = [[0, 1], [0, 1]]#padding = [[0, 0], [0, 1], [0, 1]]
    #     # y_true_res = tf.pad(y_true_sq, padding)
    #     # y_pred_res = tf.pad(y_pred_sq, padding)
    #     # print("y_true_res", y_true_res.shape)
    #     mean_train = self.dataset.mean_train
    #     std_train = self.dataset.std_train
    #     y_true_pow = tf.exp(2.0 * (y_true_sq * std_train + mean_train))
    #     y_pred_pow = tf.exp(2.0 * (y_pred_sq * std_train + mean_train))
    #     mse_loss = tf.reduce_mean(tf.square(y_pred_sq - y_true_sq), axis=1)
    #     pmsqe_loss = pmsqe.per_frame_PMSQE(y_true_pow, y_pred_pow, alpha=0.1)
    #     print('pmsqe_loss', pmsqe_loss)
    #     print('mse_loss', mse_loss)
    #     return tf.reduce_mean(pmsqe_loss + mse_loss)

    # def speechVGG_model(self,
    #                     include_top=False,
    #                     weights='/Users/kadkhodm/PycharmProjects/speech_inpainting/weights/1000hours_6000words_weights.10-0.30.h5',
    #                     input_shape=(MAX_DEC_SEQ_LEN, NUM_DEC_FEAT, 1),
    #                     classes=8,
    #                     pooling=None,
    #                     transfer_learning=False):
    #
    #     speech_model = speechVGG(include_top=include_top,
    #                              weights=weights,
    #                              input_shape=input_shape,
    #                              classes=classes,
    #                              pooling=pooling,
    #                              transfer_learning=transfer_learning)
    #
    #     return speech_model

    def build_network(self):

        if self.multimodal_flag:

            if self.multi_tasking_flag and self.seq2seq_flag and not self.attention_flag:

                self.enc_in = Input((MAX_ENC_SEQ_LEN, NUM_ENC_FEAT), name='encoder_input')
                self.enc_rnn1 = Bidirectional(LSTM(self.LATENT_DIM,
                                                   kernel_initializer='Orthogonal',
                                                   return_sequences=True, name='encoder_lstm1'),
                                              merge_mode='concat')(self.enc_in)
                self.enc_rnn2 = Bidirectional(LSTM(self.LATENT_DIM,
                                                   kernel_initializer='Orthogonal',
                                                   return_sequences=True, name='encoder_lstm2'),
                                              merge_mode='concat')(self.enc_rnn1)
                self.enc_rnn3 = Bidirectional(LSTM(self.LATENT_DIM,
                                                   kernel_initializer='Orthogonal',
                                                   return_sequences=True, name='encoder_lstm3'),
                                              merge_mode='concat')(self.enc_rnn2)

                self.enc_vis = Dense(NUM_ENC_FEAT, activation="relu")(self.enc_rnn3)
                self.algn = Dense(ALGN_LEN, kernel_initializer='he_normal',
                                  name='align', activation='softmax')(self.enc_vis)

                self.dec_in = Input(shape=(MAX_DEC_SEQ_LEN, NUM_DEC_FEAT), name='decoder_input')
                self.dec_cat = concatenate([self.dec_in, self.enc_vis], axis=-1)

                self.dec_rnn1 = Bidirectional(LSTM(self.LATENT_DIM,
                                                   return_sequences=True,
                                                   activation="tanh", name='decoder_lstm_1'))(self.dec_cat)
                self.dec_rnn2 = Bidirectional(LSTM(self.LATENT_DIM,
                                                   return_sequences=True,
                                                   activation="tanh", name='decoder_lstm_2'))(self.dec_rnn1)
                self.dec_rnn3 = Bidirectional(LSTM(self.LATENT_DIM,
                                                   return_sequences=True,
                                                   activation="tanh", name='decoder_lstm_3'))(self.dec_rnn2)

                self.dec_out = Dense(NUM_DEC_FEAT, activation="relu", name='decoder_output')(self.dec_rnn3)
                # self.dec_fc = Dense(NUM_ENC_FEAT, activation="relu", name='decoder_fc')(self.dec_rnn3)
                # self.algn = Dense(ALGN_LEN, kernel_initializer='he_normal',
                #                   name='align', activation='softmax')(self.dec_fc)

                self.model = Model([self.enc_in, self.dec_in], [self.algn, self.dec_out])

            elif self.multi_tasking_flag and self.seq2seq_flag and self.attention_flag:

                self.dec_in = Input(shape=(MAX_DEC_SEQ_LEN, NUM_DEC_FEAT), name='decoder_input')
                self.enc_in = Input(shape=(MAX_ENC_SEQ_LEN, NUM_ENC_FEAT), name='encoder_input')

                a = Reshape((MAX_DEC_SEQ_LEN, NUM_DEC_FEAT, 1), name="expand_dim")(self.dec_in)
                a = Conv2D(filters=32, kernel_size=[11, 41], strides=[1, 2],
                           padding="same", use_bias=False, name="conv_1", )(a)
                a = BatchNormalization(name="conv_1_bn")(a)
                a = ReLU(name="conv_1_relu")(a)
                a = Conv2D(filters=32, kernel_size=[11, 21], strides=[1, 2],
                           padding="same", use_bias=False, name="conv_2", )(a)
                a = BatchNormalization(name="conv_2_bn")(a)
                a = ReLU(name="conv_2_relu")(a)
                a = Reshape((-1, a.shape[-2] * a.shape[-1]))(a)
                for i in range(1, 5 + 1):
                    recurrent = GRU(units=self.LATENT_DIM, activation="tanh",
                                    recurrent_activation="sigmoid", use_bias=True,
                                    return_sequences=True, reset_after=True, name=f"agru_{i}", )
                    a = Bidirectional(recurrent, name=f"bidirectional_{i}", merge_mode="concat")(a)
                    if i < 5:
                        a = Dropout(rate=0.5)(a)
                a = Dense(units=self.LATENT_DIM * 2, name="dense_1")(a)
                a = ReLU(name="dense_1_relu")(a)
                a = Dropout(rate=0.5)(a)

                v = self.enc_in
                for i in range(1, 3 + 1):
                    elstm = LSTM(self.LATENT_DIM, kernel_initializer='Orthogonal',
                                 return_sequences=True, activation="tanh",
                                 recurrent_activation="sigmoid", name=f"vlstm_{i}", )
                    v = Bidirectional(elstm, name=f"vlstm_{i}", merge_mode="concat")(v)

                self.encs_cat = concatenate([v, a], axis=-1)
                self.algn = Dense(units=ALGN_LEN, name='align',
                                  activation="softmax")(self.encs_cat)

                self.dec_cat = concatenate([self.dec_in, self.encs_cat], axis=-1)
                s = self.dec_cat
                for i in range(1, 3 + 1):
                    dlstm = LSTM(self.LATENT_DIM, kernel_initializer='Orthogonal',
                                 return_sequences=True, activation="tanh",
                                 recurrent_activation="sigmoid", name=f"slstm_{i}", )
                    s = Bidirectional(dlstm, name=f"slstm_{i}", merge_mode="concat")(s)

                self.dec_out = Dense(NUM_DEC_FEAT, activation="relu", name='decoder_output')(s)
                self.model = Model([self.enc_in, self.dec_in], [self.algn, self.dec_out])

            # elif self.multi_tasking_flag and self.seq2seq_flag and self.attention_flag:
            #
            #     self.enc_in = Input((MAX_ENC_SEQ_LEN, NUM_ENC_FEAT), name='encoder_input')
            #     self.enc_rnn1 = Bidirectional(LSTM(self.LATENT_DIM,
            #                                        kernel_initializer='Orthogonal',
            #                                        return_sequences=True, name='encoder_lstm1'),
            #                                   merge_mode='concat')(self.enc_in)
            #     self.enc_rnn2 = Bidirectional(LSTM(self.LATENT_DIM,
            #                                        kernel_initializer='Orthogonal',
            #                                        return_sequences=True, name='encoder_lstm2'),
            #                                   merge_mode='concat')(self.enc_rnn1)
            #     self.enc_rnn3 = Bidirectional(LSTM(self.LATENT_DIM,
            #                                        kernel_initializer='Orthogonal',
            #                                        return_sequences=True, name='encoder_lstm3'),
            #                                   merge_mode='concat')(self.enc_rnn2)
            #
            #     self.enc_vis = Dense(NUM_ENC_FEAT, activation="relu")(self.enc_rnn3)
            #     self.algn = Dense(ALGN_LEN, kernel_initializer='he_normal',
            #                       name='align', activation='softmax')(self.enc_vis)
            #
            #     self.dec_in = Input(shape=(MAX_DEC_SEQ_LEN, NUM_DEC_FEAT), name='decoder_input')
            #     self.dec_cat = concatenate([self.dec_in, self.enc_rnn3, self.enc_in], axis=-1)
            #
            #     self.dec_rnn1 = Bidirectional(LSTM(self.LATENT_DIM,
            #                                        return_sequences=True,
            #                                        activation="tanh", name='decoder_lstm_1'))(self.dec_cat)
            #     self.dec_rnn2 = Bidirectional(LSTM(self.LATENT_DIM,
            #                                        return_sequences=True,
            #                                        activation="tanh", name='decoder_lstm_2'))(self.dec_rnn1)
            #     self.dec_rnn3 = Bidirectional(LSTM(self.LATENT_DIM,
            #                                        return_sequences=True,
            #                                        activation="tanh", name='decoder_lstm_3'))(self.dec_rnn2)
            #
            #     self.dec_out = Dense(NUM_DEC_FEAT, activation="relu", name='decoder_output')(self.dec_rnn3)
            #     # self.dec_fc = Dense(NUM_ENC_FEAT, activation="relu", name='decoder_fc')(self.dec_rnn3)
            #     # self.algn = Dense(ALGN_LEN, kernel_initializer='he_normal',
            #     #                   name='align', activation='softmax')(self.dec_fc)
            #
            #     self.model = Model([self.enc_in, self.dec_in], [self.algn, self.dec_out])

            elif self.multi_tasking_flag and not self.seq2seq_flag:

                self.enc_in = Input((MAX_ENC_SEQ_LEN, NUM_ENC_FEAT), name='encoder_input')
                self.dec_in = Input(shape=(MAX_DEC_SEQ_LEN, NUM_DEC_FEAT), name='decoder_input')
                self.dec_cat = concatenate([self.dec_in, self.enc_in], axis=-1)

                self.dec_rnn1 = Bidirectional(LSTM(self.LATENT_DIM,
                                                   return_sequences=True,
                                                   activation="tanh", name='decoder_lstm_1'))(self.dec_cat)
                self.dec_rnn2 = Bidirectional(LSTM(self.LATENT_DIM,
                                                   return_sequences=True,
                                                   activation="tanh", name='decoder_lstm_2'))(self.dec_rnn1)
                self.dec_rnn3 = Bidirectional(LSTM(self.LATENT_DIM,
                                                   return_sequences=True,
                                                   activation="tanh", name='decoder_lstm_3'))(self.dec_rnn2)

                self.dec_out = Dense(NUM_DEC_FEAT, activation="relu", name='decoder_output')(self.dec_rnn3)
                self.dec_fc = Dense(NUM_ENC_FEAT, activation="relu", name='decoder_fc')(self.dec_rnn3)
                self.algn = Dense(ALGN_LEN, kernel_initializer='he_normal',
                                  name='align', activation='softmax')(self.dec_fc)

                self.model = Model([self.enc_in, self.dec_in], [self.algn, self.dec_out])

            elif not self.multi_tasking_flag and self.seq2seq_flag:

                self.enc_in = Input((MAX_ENC_SEQ_LEN, NUM_ENC_FEAT), name='encoder_input')
                self.enc_rnn1 = Bidirectional(LSTM(self.LATENT_DIM,
                                                   kernel_initializer='Orthogonal',
                                                   return_sequences=True, name='encoder_lstm1'),
                                              merge_mode='concat')(self.enc_in)
                self.enc_rnn2 = Bidirectional(LSTM(self.LATENT_DIM,
                                                   kernel_initializer='Orthogonal',
                                                   return_sequences=True, name='encoder_lstm2'),
                                              merge_mode='concat')(self.enc_rnn1)
                self.enc_rnn3 = Bidirectional(LSTM(self.LATENT_DIM,
                                                   kernel_initializer='Orthogonal',
                                                   return_sequences=True, name='encoder_lstm3'),
                                              merge_mode='concat')(self.enc_rnn2)

                self.dec_in = Input(shape=(MAX_DEC_SEQ_LEN, NUM_DEC_FEAT), name='decoder_input')
                self.dec_cat = concatenate([self.dec_in, self.enc_rnn3], axis=-1)

                self.dec_rnn1 = Bidirectional(LSTM(self.LATENT_DIM,
                                                   return_sequences=True,
                                                   activation="tanh", name='decoder_lstm_1'))(self.dec_cat)
                self.dec_rnn2 = Bidirectional(LSTM(self.LATENT_DIM,
                                                   return_sequences=True,
                                                   activation="tanh", name='decoder_lstm_2'))(self.dec_rnn1)
                self.dec_rnn3 = Bidirectional(LSTM(self.LATENT_DIM,
                                                   return_sequences=True,
                                                   activation="tanh", name='decoder_lstm_3'))(self.dec_rnn2)

                self.dec_out = Dense(NUM_DEC_FEAT, activation="relu", name='decoder_output')(self.dec_rnn3)

                self.model = Model([self.enc_in, self.dec_in], self.dec_out)

            elif not self.multi_tasking_flag and not self.seq2seq_flag:

                self.enc_in = Input((MAX_ENC_SEQ_LEN, NUM_ENC_FEAT), name='encoder_input')
                self.dec_in = Input(shape=(MAX_DEC_SEQ_LEN, NUM_DEC_FEAT), name='decoder_input')
                self.dec_cat = concatenate([self.dec_in, self.enc_in], axis=-1)

                self.dec_rnn1 = Bidirectional(LSTM(self.LATENT_DIM,
                                                   return_sequences=True,
                                                   activation="tanh", name='decoder_lstm_1'))(self.dec_cat)
                self.dec_rnn2 = Bidirectional(LSTM(self.LATENT_DIM,
                                                   return_sequences=True,
                                                   activation="tanh", name='decoder_lstm_2'))(self.dec_rnn1)
                self.dec_rnn3 = Bidirectional(LSTM(self.LATENT_DIM,
                                                   return_sequences=True,
                                                   activation="tanh", name='decoder_lstm_3'))(self.dec_rnn2)

                self.dec_out = Dense(NUM_DEC_FEAT, activation="relu", name='decoder_output')(self.dec_rnn3)

                self.model = Model([self.enc_in, self.dec_in], self.dec_out)

            elif self.attention_flag:

                self.enc_in = Input((MAX_ENC_SEQ_LEN, NUM_ENC_FEAT), name='encoder_input')
                self.enc_rnn1 = Bidirectional(LSTM(self.LATENT_DIM,
                                                   kernel_initializer='Orthogonal',
                                                   return_sequences=True, name='encoder_lstm1'),
                                              merge_mode='concat')(self.enc_in)
                self.enc_rnn2 = Bidirectional(LSTM(self.LATENT_DIM,
                                                   kernel_initializer='Orthogonal',
                                                   return_sequences=True, name='encoder_lstm2'),
                                              merge_mode='concat')(self.enc_rnn1)
                self.enc_rnn3 = Bidirectional(LSTM(self.LATENT_DIM,
                                                   kernel_initializer='Orthogonal',
                                                   return_sequences=True, name='encoder_lstm3'),
                                              merge_mode='concat')(self.enc_rnn2)

                self.dec_in = Input(shape=(MAX_DEC_SEQ_LEN, NUM_DEC_FEAT), name='decoder_input')

                self.dec_rnn1 = Bidirectional(LSTM(self.LATENT_DIM,
                                                   return_sequences=True,
                                                   activation="tanh", name='decoder_lstm_1'))(self.dec_in)
                self.attn_out, self.attn_scores = MultiHeadAttention(key_dim=NUM_ENC_FEAT,
                                                                     num_heads=8)(query=self.dec_rnn1,
                                                                                  value=self.enc_rnn3,
                                                                                  return_attention_scores=True)
                self.dec_concat = concatenate([self.dec_rnn1, self.attn_out], axis=-1, name='concat_layer')
                self.dec_rnn2 = Bidirectional(LSTM(self.LATENT_DIM,
                                                   return_sequences=True,
                                                   activation="tanh", name='decoder_lstm_2'))(self.dec_concat)
                self.dec_rnn3 = Bidirectional(LSTM(self.LATENT_DIM,
                                                   return_sequences=True,
                                                   activation="tanh", name='decoder_lstm_3'))(self.dec_rnn2)

                self.dec_out = Dense(NUM_DEC_FEAT, activation="relu", name='decoder_output')(self.dec_rnn3)

                self.model = Model([self.enc_in, self.dec_in], self.dec_out)

        else:
            if self.cnn_flag:

                decoder_inputs = Input(shape=(MAX_DEC_SEQ_LEN, NUM_DEC_FEAT), name='decoder_input')
                decoder_inputs_exp = tf.expand_dims(decoder_inputs, axis=-1)

                dec_zero1 = ZeroPadding2D(padding=(1, 1), name='zero1')(decoder_inputs_exp)
                dec_conv1 = Conv2D(32, (3, 3), strides=(1, 1),
                                   kernel_initializer='he_normal', name='conv1')(dec_zero1)
                dec_batc1 = BatchNormalization(name='batc1')(dec_conv1)
                dec_actv1 = Activation('relu', name='actv1')(dec_batc1)
                dec_drop1 = SpatialDropout2D(0.5)(dec_actv1)
                dec_maxp1 = MaxPooling2D(pool_size=(1, 4), strides=(1, 4), name='max1')(dec_drop1)

                dec_zero2 = ZeroPadding2D(padding=(1, 1), name='zero2')(dec_maxp1)
                dec_conv2 = Conv2D(64, (3, 3), strides=(1, 1),
                                   kernel_initializer='he_normal', name='conv2')(dec_zero2)
                dec_batc2 = BatchNormalization(name='batc2')(dec_conv2)
                dec_actv2 = Activation('relu', name='actv2')(dec_batc2)
                dec_drop2 = SpatialDropout2D(0.5)(dec_actv2)
                dec_maxp2 = MaxPooling2D(pool_size=(1, 4), strides=(1, 4), name='max2')(dec_drop2)

                dec_zero3 = ZeroPadding2D(padding=(1, 1), name='zero3')(dec_maxp2)
                dec_conv3 = Conv2D(128, (3, 3), strides=(1, 1),
                                   kernel_initializer='he_normal', name='conv3')(dec_zero3)
                dec_batc3 = BatchNormalization(name='batc3')(dec_conv3)
                dec_actv3 = Activation('relu', name='actv3')(dec_batc3)
                dec_drop3 = SpatialDropout2D(0.5)(dec_actv3)
                dec_maxp3 = MaxPooling2D(pool_size=(1, 2), strides=(1, 2), name='max3')(dec_drop3)
                dec_resh1 = TimeDistributed(Flatten())(dec_maxp3)

                dec_rnn1 = Bidirectional(LSTM(self.LATENT_DIM, return_sequences=True,
                                              activation="tanh", name='decoder_lstm_2'))(dec_resh1)
                dec_rnn2 = Bidirectional(LSTM(self.LATENT_DIM, return_sequences=True,
                                              activation="tanh", name='decoder_lstm_3'))(dec_rnn1)
                dec_rnn3 = Bidirectional(LSTM(self.LATENT_DIM, return_sequences=True,
                                              activation="tanh", name='decoder_lstm_3'))(dec_rnn2)
                decoder_outputs = TimeDistributed(Dense(NUM_DEC_FEAT, activation="relu"))(dec_rnn3)
                """
                dec_rnn4_exp = tf.expand_dims(dec_rnn3, axis=-1)

                dec_zero4 = ZeroPadding2D(padding=(1, 1), name='zero4')(dec_rnn4_exp)
                dec_conv4 = Conv2D(64, (3, 3), strides=(1, 1),
                                   kernel_initializer='he_normal', name='conv4')(dec_zero4)
                dec_batc4 = BatchNormalization(name='batc4')(dec_conv4)
                dec_actv4 = Activation('relu', name='actv4')(dec_batc4)
                dec_drop4 = SpatialDropout2D(0.5)(dec_actv4)
                dec_maxp4 = MaxPooling2D(pool_size=(1, 4), strides=(1, 4), name='max4')(dec_drop4)

                dec_zero5 = ZeroPadding2D(padding=(1, 1), name='zero5')(dec_maxp4)
                dec_conv5 = Conv2D(32, (3, 3), strides=(1, 1),
                                   kernel_initializer='he_normal', name='conv5')(dec_zero5)
                dec_batc5 = BatchNormalization(name='batc5')(dec_conv5)
                dec_actv5 = Activation('relu', name='actv5')(dec_batc5)
                dec_drop5 = SpatialDropout2D(0.5)(dec_actv5)
                dec_maxp5 = MaxPooling2D(pool_size=(1, 4), strides=(1, 4), name='max5')(dec_drop5)

                dec_zero6 = ZeroPadding2D(padding=(1, 1), name='zero6')(dec_maxp5)
                dec_conv6 = Conv2D(16, (3, 3), strides=(1, 1),
                                   kernel_initializer='he_normal', name='conv6')(dec_zero6)
                dec_batc6 = BatchNormalization(name='batc6')(dec_conv6)
                dec_actv6 = Activation('relu', name='actv6')(dec_batc6)
                dec_drop6 = SpatialDropout2D(0.5)(dec_actv6)
                dec_maxp6 = MaxPooling2D(pool_size=(1, 2), strides=(1, 2), name='max6')(dec_drop6)
                dec_resh2 = TimeDistributed(Flatten())(dec_maxp6)

                decoder_outputs = TimeDistributed(Dense(NUM_DEC_FEAT, activation="relu"))(dec_resh2)
                """

                self.model = Model(decoder_inputs, decoder_outputs)

            else:

                self.dec_in = Input(shape=(MAX_DEC_SEQ_LEN, NUM_DEC_FEAT), name='decoder_input')
                self.dec_rnn1 = Bidirectional(LSTM(self.LATENT_DIM, return_sequences=True,
                                                   activation="tanh", name='decoder_lstm_1'))(self.dec_in)
                self.dec_rnn2 = Bidirectional(LSTM(self.LATENT_DIM, return_sequences=True,
                                                   activation="tanh", name='decoder_lstm_2'))(self.dec_rnn1)
                self.dec_rnn3 = Bidirectional(LSTM(self.LATENT_DIM, return_sequences=True,
                                                   activation="tanh", name='decoder_lstm_3'))(self.dec_rnn2)

                self.dec_out = Dense(NUM_DEC_FEAT, name='decoder_output')(self.dec_rnn3)

                self.model = Model(self.dec_in, self.dec_out)

        self.model.summary()

    def callbacks_list(self):
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=os.path.join(
            self.MODEL_LOGS_PATH,
            f'tensorboard_{self.model_name}_{self.run_name}'),
            histogram_freq=1,
            write_graph=True,
            write_images=True,
            update_freq='epoch',
            profile_batch=2,
            embeddings_freq=1,
            embeddings_metadata=None)

        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(os.path.join(
            self.MODEL_LOGS_PATH,
            f'checkpoint_{self.model_name}_{self.run_name}'),
            monitor='val_loss',
            verbose=0,
            save_best_only=True,
            save_weights_only=False,
            mode='auto',
            save_freq='epoch')

        reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss",
                                                                  factor=0.1,
                                                                  patience=5,
                                                                  verbose=0,
                                                                  mode="auto",
                                                                  min_delta=0.0001)

        earlystopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                                  min_delta=0.0001,
                                                                  patience=10,
                                                                  verbose=0,
                                                                  mode='auto',
                                                                  baseline=None,
                                                                  restore_best_weights=True)

        # sample_weights_callback = SampleWeights()
        # validation_callback = CallbackEval(self.validation_dataset)
        def log_images(epoch, logs):
            if self.multi_tasking_flag:
                (frames, masked_spec), (align, clean_spec) = self.dataset["val"].as_numpy_iterator().next()
                pred_align, pred_spec = self.model.predict([frames, masked_spec])
                wandb.log({"examples": [wandb.Image(np.hstack([data, pred_spec[i]]), caption=str(i))
                                        for i, data in enumerate(clean_spec)]}, commit=False)
            elif self.multimodal_flag:
                (frames, masked_spec), (clean_spec) = self.dataset["val"].as_numpy_iterator().next()
                pred_spec = self.model.predict([frames, masked_spec])
                wandb.log({"examples": [wandb.Image(np.hstack([data, pred_spec[i]]), caption=str(i))
                                        for i, data in enumerate(clean_spec)]}, commit=False)
            else:
                (masked_spec), (clean_spec) = self.dataset["val"].as_numpy_iterator().next()
                pred_spec = self.model.predict([masked_spec])
                wandb.log({"examples": [wandb.Image(np.hstack([data, pred_spec[i]]), caption=str(i))
                                        for i, data in enumerate(clean_spec)]}, commit=False)

            # pred_align, pred_spec = self.model.predict([np.array(self.data['test']["encoder_input"]),
            #                                             np.array(self.data['test']["decoder_input"])])
            # wandb.log({"examples": [wandb.Image(np.hstack([data, pred_spec[i]]), caption=str(i))
            #                         for i, data in enumerate(np.array(self.data['test']["decoder_target"]))]},
            #           commit=False)

        def log_alignment(epoch, logs):
            (frames, masked_spec), (align, clean_spec) = self.dataset["val"].as_numpy_iterator().next()
            pred_align, pred_spec = self.model.predict([frames, masked_spec])
            input_len = np.ones(pred_align.shape[0]) * pred_align.shape[1]
            decoded = tf.keras.backend.ctc_decode(pred_align, input_length=input_len, greedy=False)[0][0]
            for x in range(len(pred_align)):
                print('Original:', tf.strings.reduce_join(num_to_char(align[x])).numpy().decode('utf-8'))
                print('Prediction:', tf.strings.reduce_join(num_to_char(decoded[x])).numpy().decode('utf-8'))
                print('~' * 100)
            # pred_align, pred_spec = self.model.predict([np.array(self.data['test']["encoder_input"]),
            #                                             np.array(self.data['test']["decoder_input"])])
            # input_len = np.ones(pred_align.shape[0]) * pred_align.shape[1]
            # decoded = tf.keras.backend.ctc_decode(pred_align, input_length=input_len, greedy=False)[0][0]
            # for x in range(len(pred_align)):
            #     print('Original:', tf.strings.reduce_join(num_to_char(np.array(self.data['test']["align"])[x])).numpy().decode('utf-8'))
            #     print('Prediction:', tf.strings.reduce_join(num_to_char(decoded[x])).numpy().decode('utf-8'))
            #     print('~' * 100)

        if self.multi_tasking_flag:
            callbacks_list = [tensorboard_callback,
                              checkpoint_callback,
                              reduce_lr_callback,
                              earlystopping_callback,
                              WandbCallback(), WandbMetricsLogger(), WandbModelCheckpoint("models"),
                              LambdaCallback(on_epoch_end=log_images), LambdaCallback(on_epoch_end=log_alignment)
                              ]
        else:
            callbacks_list = [tensorboard_callback,
                              checkpoint_callback,
                              reduce_lr_callback,
                              earlystopping_callback,
                              WandbCallback(), WandbMetricsLogger(), WandbModelCheckpoint("models"),
                              LambdaCallback(on_epoch_end=log_images)]
        # sample_weights_callback,]
        # validation_callback]

        return callbacks_list

    def train_model(self):

        self.ds.load_data(split='train', augmented=True)
        self.ds.load_data(split='val', augmented=True)
        self.dataset['train'] = self.ds.dataset['train']
        self.dataset['val'] = self.ds.dataset['val']
        callbacks = self.callbacks_list()
        opt = Adam(learning_rate=self.LEARNING_RATE)
        self.model.compile(loss=self.loss_func,
                           loss_weights=self.loss_weights,
                           optimizer=opt)
        self.model.fit(self.dataset['train'],  #self.x_train, self.y_train, #
                       # sample_weight=self.sample_weight,
                       batch_size=self.BATCH_SIZE,
                       epochs=EPOCHS,
                       validation_data=self.dataset['val'],  #(self.x_val, self.y_val),#
                       callbacks=callbacks,
                       shuffle=True)
        wandb.log({"loss": self.loss_func})
        output_dir = os.path.join(self.SAVED_MODEL_PATH, f'{self.model_name}_{str(self.run_name)}')
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
            print(output_dir)
        self.model.save(output_dir)

    def resume_train_model(self, model_path):

        self.ds.load_data(split='train', augmented=True)
        self.ds.load_data(split='val', augmented=True)
        self.dataset['train'] = self.ds.dataset['train']
        self.dataset['val'] = self.ds.dataset['val']
        callbacks = self.callbacks_list()
        if self.multi_tasking_flag:
            self.model = load_model(model_path, custom_objects={'loss_func': self.loss_func,
                                                                'ctc_loss': self.ctc_loss,
                                                                'l2_loss': self.l2_loss}, compile=True)
        else:
            self.model = load_model(model_path, custom_objects={'loss_func': self.loss_func,
                                                                'l2_loss': self.l2_loss}, compile=True)
        # custom_objects={'custom_loss': self.loss_func}, compile=True)

        self.model.fit(self.dataset['train'], #self.x_train, self.y_train, #
                       # sample_weight=self.sample_weight,
                       batch_size=self.BATCH_SIZE,
                       epochs=EPOCHS,
                       validation_data=self.dataset['val'], #(self.x_val, self.y_val), #
                       callbacks=callbacks,
                       shuffle=True)

    def pred_model(self, split):

        self.ds.load_data(split=split, augmented=False)
        self.dataset[split] = self.ds.dataset[split]
        if self.multi_tasking_flag:
            self.align[split], self.pred[split] = self.model.predict(self.dataset[split])
        else:
            self.pred[split] = self.model.predict(self.dataset[split])
        # if self.multi_tasking_flag:
        #     self.align[split], self.pred[split] = self.model.predict([np.array(self.data[split]["encoder_input"]),
        #                                                               np.array(self.data[split]["decoder_input"])])
        # else:
        #     self.pred[split] = self.model.predict([np.array(self.data[split]["encoder_input"]),
        #                                            np.array(self.data[split]["decoder_input"])])

        # mask = [np.where(self.data[split]["decoder_mask"][x] == 0, 1, 0)
        #         for x in range(0, len(self.data[split]["decoder_mask"]))]
        # self.y_res[split] = self.pred[split] * np.array(mask)
        # self.y_hat[split] = np.array(self.data[split]["decoder_input"]) + self.y_res[split]

    def load_model(self, model_path, split):

        self.model = load_model(model_path, compile=False)
        self.pred_model(split=split)
        self.spec2audio(split=split)

    def spec2audio(self, split):

        storage_path = os.path.join(self.AUDIO_RESULTS_PATH, f'{self.model_name}_{split}_{self.run_name}')
        if not os.path.isdir(storage_path):
            os.makedirs(storage_path)
            print("created folder : ", storage_path)
        if not self.multi_tasking_flag:
            # phone_model = load_model(
            #     '/Users/kadkhodm/Desktop/Research/inpainting/logs/phone_recognizer_resumed_keras_checkpoint',
            #     compile=False)
            # self.align[split] = np.array(phone_model.predict(np.array(self.pred[split])))
            avg_pesq, avg_stoi, avg_psnr, avg_ssim, avg_mse = self.ds.spec2audio(split=split,
                                                                                 dec_pred=np.array(self.pred[split]),
                                                                                 algn_pred=None,
                                                                                 output_dir=storage_path,
                                                                                 sample_rate=self.sample_rate,
                                                                                 save_audio=True,
                                                                                 model_name=self.model_name,
                                                                                 informed=False)
        else:
            avg_pesq, avg_stoi, avg_psnr, avg_ssim, avg_mse = self.ds.spec2audio(split=split,
                                                                                 dec_pred=np.array(self.pred[split]),
                                                                                 algn_pred=np.array(self.align[split]),
                                                                                 output_dir=storage_path,
                                                                                 sample_rate=self.sample_rate,
                                                                                 save_audio=True,
                                                                                 model_name=self.model_name,
                                                                                 informed=False)

        return avg_pesq, avg_stoi, avg_psnr, avg_ssim, avg_mse

    # Hyperparameter Tuning:
    def objective(self, trial):

        LATENT_DIM = 256
        BATCH_SIZE = 16  # trial.suggest_int('batch_size', 16, 64)
        # activation = trial.suggest_categorical('activation', ['relu', 'sigmoid'])
        LEARNING_RATE = trial.suggest_loguniform('learning_rate', 1e-4, 1e-3)
        reduce_lr_patience = trial.suggest_int('reduce_lr_patience', 5, 10)
        earlystopping_patience = trial.suggest_int('reduce_lr_patience', 10, 15)

        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=os.path.join(
            self.MODEL_LOGS_PATH,
            f'tensorboard_{self.model_name}_{self.run_name}'),
            histogram_freq=1,
            write_graph=True,
            write_images=True,
            update_freq='epoch',
            profile_batch=2,
            embeddings_freq=1,
            embeddings_metadata=None)

        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(os.path.join(
            self.MODEL_LOGS_PATH,
            f'checkpoint_{self.model_name}_{self.run_name}'),
            monitor='val_loss',
            verbose=0,
            save_best_only=True,
            save_weights_only=False,
            mode='auto',
            save_freq='epoch')

        reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss",
                                                                  factor=0.1,
                                                                  patience=reduce_lr_patience,
                                                                  verbose=0,
                                                                  mode="auto",
                                                                  min_delta=0.00001)

        earlystopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                                  min_delta=0.00001,
                                                                  patience=earlystopping_patience,
                                                                  verbose=0,
                                                                  mode='auto',
                                                                  baseline=None,
                                                                  restore_best_weights=True)
        callbacks_list = [tensorboard_callback,
                          checkpoint_callback,
                          reduce_lr_callback,
                          earlystopping_callback, ]

        def create_model():
            enc_in = Input((MAX_ENC_SEQ_LEN, NUM_ENC_FEAT), name='encoder_input')
            enc_rnn1 = Bidirectional(LSTM(LATENT_DIM,
                                          kernel_initializer='Orthogonal',
                                          return_sequences=True, name='encoder_lstm1'),
                                     merge_mode='concat')(enc_in)
            enc_rnn2 = Bidirectional(LSTM(LATENT_DIM,
                                          kernel_initializer='Orthogonal',
                                          return_sequences=True, name='encoder_lstm2'),
                                     merge_mode='concat')(enc_rnn1)
            enc_rnn3 = Bidirectional(LSTM(LATENT_DIM,
                                          kernel_initializer='Orthogonal',
                                          return_sequences=True, name='encoder_lstm3'),
                                     merge_mode='concat')(enc_rnn2)

            enc_vis = Dense(NUM_ENC_FEAT, activation='relu')(enc_rnn3)
            algn = Dense(ALGN_LEN, kernel_initializer='he_normal',
                         name='align', activation='softmax')(enc_vis)

            dec_in = Input(shape=(MAX_DEC_SEQ_LEN, NUM_DEC_FEAT), name='decoder_input')
            dec_cat = concatenate([dec_in, enc_rnn3], axis=-1)

            dec_rnn1 = Bidirectional(LSTM(LATENT_DIM,
                                          return_sequences=True,
                                          activation="tanh", name='decoder_lstm_1'))(dec_cat)
            dec_rnn2 = Bidirectional(LSTM(LATENT_DIM,
                                          return_sequences=True,
                                          activation="tanh", name='decoder_lstm_2'))(dec_rnn1)
            dec_rnn3 = Bidirectional(LSTM(LATENT_DIM,
                                          return_sequences=True,
                                          activation="tanh", name='decoder_lstm_3'))(dec_rnn2)

            dec_out = Dense(NUM_DEC_FEAT, activation='relu',
                            name='decoder_output')(dec_rnn3)

            model = Model([enc_in, dec_in], [algn, dec_out])

            return model

        model = create_model()
        opt = Adam(learning_rate=LEARNING_RATE)
        model.compile(loss=self.loss_func,
                      loss_weights=self.loss_weights,
                      optimizer=opt)
        model.fit(self.dataset['train'],
                  batch_size=BATCH_SIZE,
                  epochs=50,
                  validation_data=self.dataset['val'],
                  callbacks=callbacks_list,
                  shuffle=True)
        sen, pred = model.predict(self.dataset['test'])

        loss = tf.reduce_mean(tf.abs(
            np.array(self.data['test']["decoder_target"]) - pred))

        return loss

    def save_results(self, split):

        storage_path = os.path.join(self.SPEC_RESULTS_PATH, f'{self.model_name}_{split}_{self.run_name}')
        if not os.path.isdir(storage_path):
            os.makedirs(storage_path)
            print("created folder : ", storage_path)

        with open(storage_path + '/' + f'pred_{self.model_name}_{split}.pkl', 'wb') as result_file:
            pickle.dump(np.array(self.pred[split]), result_file)
        # with open(storage_path + '/' + f'y_hat_{self.model_name}_{split}.pkl', 'wb') as y_hat_file:
        #     pickle.dump(np.array(self.y_hat[split]), y_hat_file)

    def plot_results(self, split, idx):

        plt.subplot(2, 3, 1)
        plt.title(f"decoder input of {split}")
        plt.imshow(self.data[split]["decoder_input"][idx, :, :])
        plt.subplot(2, 3, 2)
        plt.title(f"decoder mask of {split}")
        plt.imshow(self.data[split]["decoder_mask"][idx, :, :])

        plt.subplot(2, 3, 3)
        plt.title(f"Raw decoder output of {split}")
        plt.imshow(self.pred[split][idx, :, :])

        plt.subplot(2, 3, 4)
        plt.title(f"decoder masked output of {split}")
        plt.imshow(self.y_res[split][idx, :, :])

        # plt.subplot(2, 3, 5)
        # plt.title(f"Final output of {split}")
        # plt.imshow(self.y_hat[split][idx, :, :])

        plt.subplot(2, 3, 6)
        plt.title(f"Ground Truth of {split}")
        plt.imshow(self.data[split]["decoder_target"][idx, :, :])

        plt.show()
