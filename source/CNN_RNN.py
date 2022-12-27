
"""
CNN_RNN with bidirectional LSTMs
DLib is added!
No zero padding in spectrograms.
RepeatedVector and TimeDistributed Layers are added
Masking 10% of audio waveform, then taking the spectrogram.
three Stacked layers of biLSTM for both encoder and decoder nets
Reducing latent dims
Encoder context to all decoder inputs
batch normalization is added!
Residual connections.
"""
import os
import numpy as np
import tensorflow as tf
import tensorflow.keras
import pandas as pd
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import Input, BatchNormalization, concatenate, Dense, LSTM, Bidirectional
from tensorflow.keras.layers import add, Lambda, MultiHeadAttention, Reshape, GRU, RepeatVector, TimeDistributed
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import mse, mae
import datetime
from CNN_RNN_Dataset import Dataset
import pickle
from matplotlib import pyplot as plt
import librosa

MAX_ENC_SEQ_LEN = 75
NUM_ENC_FEAT = 32 #2048 #28
NUM_DEC_FEAT = 128
MAX_DEC_SEQ_LEN = 373

BATCH_SIZE = 64
EPOCHS = 200
LEARNING_RATE = 0.001
LATENT_DIM = 128
MISS_WGHT = 4 #1
NON_MISS_WGHT = 1

def scale_minmax(x, min=-4):

    return (x + (-1 * min)) / (-1 * min)


def descale_minmax(x_scaled, min=-4):

    return (x_scaled * (-1 * min)) + min

def make_residual_lstm_layers(input, rnn_width, rnn_depth, rnn_dropout, return_sequences):
    """
    The intermediate LSTM layers return sequences, while the last returns a single element.
    The input is also a sequence. In order to match the shape of input and output of the LSTM
    to sum them we can do it only for all layers but the last.
    """
    x = input
    for i in range(rnn_depth):
        #return_sequences = i < rnn_depth - 1
        x_rnn = LSTM(rnn_width, recurrent_dropout=rnn_dropout, dropout=rnn_dropout, return_sequences=return_sequences)(x)
        if return_sequences:
            # Intermediate layers return sequences, input is also a sequence.
            #if i > 0 or input.shape[-1] == rnn_width:
                x = add([x, x_rnn])
            #else:
                # Note that the input size and RNN output has to match, due to the sum operation.
                # If we want different rnn_width, we'd have to perform the sum from layer 2 on.
             #   x = x_rnn
        else:
            # Last layer does not return sequences, just the last element
            # so we select only the last element of the previous output.
            def slice_last(x):
                return x[..., -1, :]
            x = add([Lambda(slice_last)(x), x_rnn])
    return x


class Seq2seq():

    def __init__(self, lipnet_model_path, multimodal_flag, attention_flag, seq2seq_flag, lipnet_flag, cnn_flag):
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
        self.cnn_flag = cnn_flag
        if self.multimodal_flag:
            self.model_name = "multimodal"
        else:
            self.model_name = "audio_only"
        self.accuracy = {}
        self.pred = {}
        self.y_res = {}
        self.y_hat = {}
        self.data_mask = {}
        self.data_mask_not = {}
        self.build_network()
        self.dataset = Dataset(self.lipnet_model_path, self.multimodal_flag, self.lipnet_flag)
        self.data = self.dataset.data
        self.decoder_max = 167.2503
        self.df = {"train": pd.read_csv("train_data.csv"),
                   "test": pd.read_csv("test_data.csv")}
        # Building the train dataset during the class initialization
        # self.data = {}

    def masked_loss(self, y_true, y_pred):
        # Calculate the loss for each item in the batch.
        loss_fn = tf.keras.losses.mae
        loss = loss_fn(y_true, y_pred)
        print("loss", loss.shape)
        # Mask off the losses on padding.
        mask = tf.cast(y_true != 0, loss.dtype)
        print("mask", mask.shape)
        loss *= mask

        # Return the total.
        return tf.reduce_sum(loss) / tf.reduce_sum(mask)

    def custom_loss(self, y_true, y_pred):
        # Calculate the loss for each item in the batch.
        loss_fn = tf.keras.losses.mae
        true_mfcc = librosa.feature.mfcc(S=y_true.numpy())
        pred_mfcc = librosa.feature.mfcc(S=y_pred.numpy())

        return loss_fn(true_mfcc, pred_mfcc)

    def masked_acc(self, y_true, y_pred):
        # Calculate the loss for each item in the batch.
        # y_pred = tf.argmax(y_pred, axis=-1)
        y_pred = tf.cast(y_pred, y_true.dtype)

        match = tf.cast(y_true == y_pred, tf.float32)
        mask = tf.cast(y_true != 0, tf.float32)

        return tf.reduce_sum(match) / tf.reduce_sum(mask)

    def masked_relative_err(self, y_true, y_pred):

        y_pred = tf.cast(y_pred, y_true.dtype)
        y_true_norm = tf.math.reduce_euclidean_norm(y_true, [1, 2])
        dist_norm = tf.math.reduce_euclidean_norm(y_true-y_pred, [1, 2])
        rel_err = tf.math.divide(dist_norm, y_true_norm)

        return tf.reduce_sum(rel_err) / tf.size(rel_err)

    def build_network(self):

        if self.multimodal_flag:

            if self.attention_flag:
                encoder_inputs = Input((MAX_ENC_SEQ_LEN, NUM_ENC_FEAT), name='encoder_input')
                enc_bn1 = BatchNormalization()(encoder_inputs)
                enc_rnn1 = Bidirectional(LSTM(1 * LATENT_DIM, return_sequences=True, name='encoder_lstm_1'))(enc_bn1)
                enc_bn2 = BatchNormalization()(enc_rnn1)
                enc_rnn2 = Bidirectional(LSTM(1 * LATENT_DIM, return_sequences=True, name='encoder_lstm_2'))(enc_bn2)
                enc_bn3 = BatchNormalization()(enc_rnn2)
                enc_out = Bidirectional(LSTM(1 * LATENT_DIM, return_sequences=True, name='encoder_lstm_3'))(enc_bn3)
                # enc_out, state_h_f, state_h_b, state_c_f, state_c_b = Bidirectional(
                #    LSTM(1 * LATENT_DIM, return_sequences=True, return_state=True, name='encoder_lstm_3'))(enc_bn3)
                # encoder_states = [state_h_f, state_h_b, state_c_f, state_c_b]

                decoder_inputs = Input(shape=(MAX_DEC_SEQ_LEN, NUM_DEC_FEAT), name='decoder_input')
                dec_bn1 = BatchNormalization()(decoder_inputs)
                dec_rnn1 = Bidirectional(LSTM(1 * LATENT_DIM, return_sequences=True, name='decoder_lstm_1'))(dec_bn1)#, initial_state=encoder_states)
                dec_bn2 = BatchNormalization()(dec_rnn1)
                dec_rnn2 = Bidirectional(LSTM(1 * LATENT_DIM, return_sequences=True, name='decoder_lstm_2'))(dec_bn2)
                dec_bn3 = BatchNormalization()(dec_rnn2)
                dec_out = Bidirectional(LSTM(1 * LATENT_DIM, return_sequences=True, name='decoder_lstm_3'))(dec_bn3)

                attn_out, attn_scores = MultiHeadAttention(key_dim=NUM_DEC_FEAT, num_heads=4)(query=dec_out,
                                                                                              value=enc_out,
                                                                                              return_attention_scores=True)
                add = tf.keras.layers.Add()([dec_out, attn_out])
                layernorm = tf.keras.layers.LayerNormalization()(add)
                # dec_concat_out = concatenate([dec_out, attn_out], axis=-1, name='concat_layer')
                dec_pred = TimeDistributed(Dense(NUM_DEC_FEAT))(layernorm)

                self.model = Model([encoder_inputs, decoder_inputs], outputs=dec_pred)

            elif self.seq2seq_flag:
                encoder_inputs = Input((MAX_ENC_SEQ_LEN, NUM_ENC_FEAT), name='encoder_input')
                enc_bn1 = BatchNormalization()(encoder_inputs)
                enc_rnn1 = Bidirectional(LSTM(1 * LATENT_DIM, return_sequences=True, name='encoder_lstm_1'))(enc_bn1)
                enc_bn2 = BatchNormalization()(enc_rnn1)
                enc_rnn2 = Bidirectional(LSTM(1 * LATENT_DIM, return_sequences=True, name='encoder_lstm_2'))(enc_bn2)
                enc_bn3 = BatchNormalization()(enc_rnn2)
                enc_rnn3 = Bidirectional(LSTM(1 * LATENT_DIM, return_sequences=False, name='encoder_lstm_3'))(enc_bn3)
                # BN_enc_lay3 = BatchNormalization()(encoder_lay3)
                repeated_enc_out = RepeatVector(MAX_DEC_SEQ_LEN)(enc_rnn3)

                decoder_inputs_1 = Input(shape=(MAX_DEC_SEQ_LEN, NUM_DEC_FEAT), name='decoder_input')
                decoder_inputs_2 = repeated_enc_out
                decoder_inputs = concatenate([decoder_inputs_1, decoder_inputs_2], axis=-1)
                dec_bn1 = BatchNormalization()(decoder_inputs)
                dec_rnn1 = Bidirectional(LSTM(1 * LATENT_DIM, return_sequences=True, name='decoder_lstm_1'))(dec_bn1)
                dec_bn2 = BatchNormalization()(dec_rnn1)
                dec_rnn2 = Bidirectional(LSTM(1 * LATENT_DIM, return_sequences=True, name='decoder_lstm_2'))(dec_bn2)
                dec_bn3 = BatchNormalization()(dec_rnn2)
                dec_rnn3 = Bidirectional(LSTM(1 * LATENT_DIM, return_sequences=True, name='decoder_lstm_3'))(dec_bn3)
                # decoder_inputs_BN4 = BatchNormalization()(decoder_lay3)
                decoder_outputs = TimeDistributed(Dense(NUM_DEC_FEAT))(dec_rnn3)

                self.model = Model([encoder_inputs, decoder_inputs_1], decoder_outputs)
            else:
                encoder_inputs = Input((MAX_ENC_SEQ_LEN, NUM_ENC_FEAT), name='encoder_input')
                decoder_inputs_1 = Input(shape=(MAX_DEC_SEQ_LEN, NUM_DEC_FEAT), name='decoder_input')
                decoder_inputs_2 = tf.repeat(encoder_inputs, np.round(MAX_DEC_SEQ_LEN / MAX_ENC_SEQ_LEN), axis=1)
                decoder_inputs = concatenate([decoder_inputs_1, decoder_inputs_2[:, :MAX_DEC_SEQ_LEN, :]], axis=-1)
                dec_bn1 = BatchNormalization()(decoder_inputs)
                dec_rnn1 = Bidirectional(LSTM(1 * LATENT_DIM, return_sequences=True, name='decoder_lstm_1'))(dec_bn1)
                dec_bn2 = BatchNormalization()(dec_rnn1)
                dec_rnn2 = Bidirectional(LSTM(1 * LATENT_DIM, return_sequences=True, name='decoder_lstm_2'))(dec_bn2)
                dec_bn3 = BatchNormalization()(dec_rnn2)
                dec_rnn3 = Bidirectional(LSTM(1 * LATENT_DIM, return_sequences=True, name='decoder_lstm_3'))(dec_bn3)
                # decoder_inputs_BN4 = BatchNormalization()(decoder_lay3)
                decoder_outputs = TimeDistributed(Dense(NUM_DEC_FEAT))(dec_rnn3)

                self.model = Model([encoder_inputs, decoder_inputs_1], decoder_outputs)

        else:
            if not self.cnn_flag:
                decoder_inputs = Input(shape=(MAX_DEC_SEQ_LEN, NUM_DEC_FEAT), name='decoder_input')
                dec_bn1 = BatchNormalization()(decoder_inputs)
                dec_rnn1 = Bidirectional(LSTM(1 * LATENT_DIM, return_sequences=True, name='decoder_lstm_1'))(dec_bn1)
                dec_bn2 = BatchNormalization()(dec_rnn1)
                dec_rnn2 = Bidirectional(LSTM(1 * LATENT_DIM, return_sequences=True, name='decoder_lstm_2'))(dec_bn2)
                dec_bn3 = BatchNormalization()(dec_rnn2)
                dec_rnn3 = Bidirectional(LSTM(1 * LATENT_DIM, return_sequences=True, name='decoder_lstm_3'))(dec_bn3)
                # decoder_inputs_BN4 = BatchNormalization()(decoder_lay3)
                decoder_outputs = TimeDistributed(Dense(NUM_DEC_FEAT))(dec_rnn3)

            # else:

            #    decoder_inputs = Input(shape=(MAX_DEC_SEQ_LEN, NUM_DEC_FEAT), name='decoder_input')
            #    dec_bn1 = BatchNormalization()(decoder_inputs)
            #    dec_rnn1 = ConvLSTM1D(1 * LATENT_DIM, return_sequences=True, name='decoder_lstm_1')(dec_bn1)
            #    dec_rnn2 = ConvLSTM1D(1 * LATENT_DIM, return_sequences=True, name='decoder_lstm_2')(dec_rnn1)
            #    dec_rnn3 = ConvLSTM1D(1 * LATENT_DIM, return_sequences=True, name='decoder_lstm_3')(dec_rnn2)
            #    # decoder_inputs_BN4 = BatchNormalization()(decoder_lay3)
            #    decoder_outputs = TimeDistributed(Dense(NUM_DEC_FEAT))(dec_rnn3)

            self.model = Model(decoder_inputs, decoder_outputs)

        # opt = Adam(learning_rate=LEARNING_RATE)
        # self.model.compile(optimizer=opt,
        #                   loss=self.masked_loss,
        #                   metrics=[self.masked_acc, self.masked_loss])
        # self.model.compile(optimizer=opt, loss=mse)

    def train_model(self):

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
                                                                  patience=50,
                                                                  verbose=0,
                                                                  mode="auto",
                                                                  min_delta=0.0001)

        # earlystopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
        #                                                          min_delta=0.0001,
        #                                                          patience=10,
        #                                                          verbose=0,
        #                                                          mode='auto',
        #                                                          baseline=None,
        #                                                          restore_best_weights=False)

        callbacks_list = [tensorboard_callback,
                          checkpoint_callback,
                          reduce_lr_callback,]
                          #earlystopping_callback]

        self.data_mask['train'] = [np.where(self.data['train']["decoder_mask"][x][:, 0] == 0, MISS_WGHT, NON_MISS_WGHT)
                                   for x in range(0, len(self.data['train']["decoder_mask"]))]
        self.data_mask_not['train'] = np.array([np.where(self.data['train']["decoder_mask"][x] == 0, 1, 0)
                                                for x in range(0, len(self.data['train']["decoder_mask"]))])
        # print(f'shape of encoder input {np.array(self.data["train"]["encoder_input"]).shape}')
        # print(f'shape of decoder input {np.array(self.data["train"]["decoder_input"]).shape}')
        # print(f'shape of decoder target {np.array(self.data["train"]["decoder_target"]).shape}')
        # print(f'shape of mask {np.array(self.data_mask["train"]).shape}')

        opt = Adam(learning_rate=LEARNING_RATE)
        self.model.compile(optimizer=opt,
                           loss=mse)# self.custom_loss,)
                           # metrics=[self.masked_relative_err])#[self.masked_acc, self.masked_loss])
        # self.model.compile(optimizer=opt, loss=mse)
        if self.multimodal_flag:
            self.model.fit([np.array(self.data['train']["encoder_input"]),
                            np.array(self.data['train']["decoder_input"])/self.decoder_max],
                           (np.array(self.data['train']["decoder_target"])/self.decoder_max),#*self.data_mask_not['train'],
                           sample_weight=np.array(self.data_mask['train']),
                           batch_size=BATCH_SIZE,
                           epochs=EPOCHS,
                           validation_split=0.2,
                           callbacks=callbacks_list,
                           shuffle=True)

        else:
            self.model.fit(np.array(self.data['train']["decoder_input"])/self.decoder_max,
                           np.array(self.data['train']["decoder_target"])/self.decoder_max,
                           sample_weight=np.array(self.data_mask['train']),
                           batch_size=BATCH_SIZE,
                           epochs=EPOCHS,
                           validation_split=0.2,
                           callbacks=callbacks_list,
                           shuffle=True)

        output_dir = os.path.join(self.SAVED_MODEL_PATH, f'seq2seq_{self.model_name}_{str(self.run_name)}')
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
            print(output_dir)
        self.model.save(output_dir)

    def evaluate_model(self, split):
        self.data_mask_not['test'] = [np.where(self.data['test']["decoder_mask"][x] == 0, 1, 0)
                                       for x in range(0, len(self.data['test']["decoder_mask"]))]
        if self.multimodal_flag:

            self.accuracy[split] = self.model.evaluate([np.array(self.data[split]["encoder_input"]),
                                                        np.array(self.data[split]["decoder_input"])/self.decoder_max],
                                                       (np.array(self.data[split]["decoder_target"])/self.decoder_max))#*self.data_mask_not['test'])
        else:
            self.accuracy[split] = self.model.evaluate(np.array(self.data[split]["decoder_input"])/self.decoder_max,
                                                       (np.array(self.data[split]["decoder_target"])/self.decoder_max))#*self.data_mask_not['test'])


    def pred_model(self, split):

        if self.multimodal_flag:

            self.pred[split] = self.model.predict([np.array(self.data[split]["encoder_input"]),
                                                   np.array(self.data[split]["decoder_input"])/self.decoder_max])
        else:
            self.pred[split] = self.model.predict(np.array(self.data[split]["decoder_input"])/self.decoder_max)

        mask = [np.where(self.data[split]["decoder_mask"][x] == 0, 1, 0)
                for x in range(0, len(self.data[split]["decoder_mask"]))]
        self.y_res[split] = self.pred[split] * self.decoder_max * np.array(mask)
        self.y_hat[split] = np.array(self.data[split]["decoder_input"]) + self.y_res[split]
        print(f"y hat shape {np.array(self.y_hat[split]).shape}")

    def spec2audio(self, split):

        storage_path = os.path.join(self.AUDIO_RESULTS_PATH, f'{self.model_name}_{split}_{self.run_name}')
        if not os.path.isdir(storage_path):
            os.makedirs(storage_path)
            print("created folder : ", storage_path)
        avg_pesq, avg_stoi = self.dataset.spec2audio(split=split,
                                                     results=np.array(self.y_hat[split]),
                                                     input=np.array(self.data[split]["decoder_input"]),
                                                     truth=np.array(self.data[split]["decoder_target"]),
                                                     name=np.array(self.data[split]["names"]),
                                                     output_dir=storage_path)

        return avg_pesq, avg_stoi

    def save_results(self, split):

        storage_path = os.path.join(self.SPEC_RESULTS_PATH, f'{self.model_name}_{split}_{self.run_name}')
        if not os.path.isdir(storage_path):
            os.makedirs(storage_path)
            print("created folder : ", storage_path)

        with open(storage_path +'/'+ f'_pred_{self.model_name}_{split}.pkl', 'wb') as result_file:
            pickle.dump(np.array(self.pred[split]), result_file)
        with open(storage_path +'/'+ f'_y_hat_{self.model_name}_{split}.pkl', 'wb') as y_hat_file:
            pickle.dump(np.array(self.y_hat[split]), y_hat_file)

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

        plt.subplot(2, 3, 5)
        plt.title(f"Final output of {split}")
        plt.imshow(self.y_hat[split][idx, :, :])

        plt.subplot(2, 3, 6)
        plt.title(f"Ground Truth of {split}")
        plt.imshow(self.data[split]["decoder_target"][idx, :, :])

        plt.show()

