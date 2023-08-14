import os
import numpy as np
import tensorflow as tf
import tensorflow.keras
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input, BatchNormalization, concatenate, Dense, LSTM, Bidirectional
from tensorflow.keras.layers import Dense, Activation, SpatialDropout3D, Flatten, ReLU, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import datetime
import pickle
from matplotlib import pyplot as plt
from CNN_RNN_Dataset import Dataset
import wandb
from wandb.keras import WandbCallback, WandbMetricsLogger, WandbModelCheckpoint, WandbEvalCallback
from tensorflow.keras.callbacks import LambdaCallback

run = wandb.init(
    # set the wandb project where this run will be logged
    project="AV-SI",

    # track hyperparameters and run metadata
    config={
        "learning_rate": 1e-3,
        "architecture": "LSTM_ENC_DEC",
        "dataset": "GRID",
        "epochs": 200,
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


class Seq2seq:

    def __init__(self, multi_tasking_flag, multimodal_flag, seq2seq_flag, landmark_flag,):

        self.model = None
        self.CURRENT_PATH = '../'
        self.AUDIO_RESULTS_PATH = os.path.join(self.CURRENT_PATH, 'audio_results')
        self.SAVED_MODEL_PATH = os.path.join(self.CURRENT_PATH, 'saved_models')
        self.MODEL_LOGS_PATH = os.path.join(self.CURRENT_PATH, 'logs')
        self.SPEC_RESULTS_PATH = os.path.join(self.CURRENT_PATH, 'spec_results')
        self.run_name = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        self.multimodal_flag = multimodal_flag
        self.seq2seq_flag = seq2seq_flag
        self.landmark_flag = landmark_flag
        self.multi_tasking_flag = multi_tasking_flag
        self.BATCH_SIZE = 32
        self.LEARNING_RATE = 1e-3
        self.LATENT_DIM = 256
        self.sample_rate = 8000
        self.pred = {}
        self.align = {}
        self.y_res = {}
        self.data_mask = {}
        self.data_mask_not = {}
        self.build_network()
        # Building the train dataset during the class initialization
        self.ds = Dataset(self.multimodal_flag, self.landmark_flag, self.multi_tasking_flag)
        self.dataset = self.ds.dataset
        self.data = self.ds.data
        self.df = {"train": pd.read_csv("train_data.csv"),
                   "val": pd.read_csv("val_data.csv"),
                   "test": pd.read_csv("test_data.csv")}
        if self.multi_tasking_flag:
            self.loss_func = {'align': self.ctc_loss, 'decoder_output': self.l2_loss}
            self.loss_weights = {'align': 0.001, 'decoder_output': 1}
        else:
            self.loss_func = {'decoder_output': self.l2_loss}
            self.loss_weights = {'decoder_output': 1}

        if self.multimodal_flag and not self.seq2seq_flag and not self.multi_tasking_flag:
            self.model_name = "AV_SI_landmarks_mse"
        elif self.multimodal_flag and self.seq2seq_flag and not self.multi_tasking_flag:
            self.model_name = "AV_S2S_landmarks_mse"
        elif self.multimodal_flag and self.seq2seq_flag and self.multi_tasking_flag:
            self.model_name = "AV_MTL_S2S_landmarks_mse"
        elif self.multimodal_flag and not self.seq2seq_flag and self.multi_tasking_flag:
            self.model_name = "AV_MTL_SI_landmarks_mse"
        elif not self.multimodal_flag:
            self.model_name = "A_SI_mse"

    def ctc_loss(self, y_true, y_pred):

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

    def build_network(self):

        if self.multimodal_flag:

            if self.multi_tasking_flag and self.seq2seq_flag:

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

                self.model = Model([self.enc_in, self.dec_in], [self.algn, self.dec_out])

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

        def log_alignment(epoch, logs):
            (frames, masked_spec), (align, clean_spec) = self.dataset["val"].as_numpy_iterator().next()
            pred_align, pred_spec = self.model.predict([frames, masked_spec])
            input_len = np.ones(pred_align.shape[0]) * pred_align.shape[1]
            decoded = tf.keras.backend.ctc_decode(pred_align, input_length=input_len, greedy=False)[0][0]
            for x in range(len(pred_align)):
                print('Original:', tf.strings.reduce_join(num_to_char(align[x])).numpy().decode('utf-8'))
                print('Prediction:', tf.strings.reduce_join(num_to_char(decoded[x])).numpy().decode('utf-8'))
                print('~' * 100)

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

        return callbacks_list

    def train_model(self):

        self.ds.load_data(split='train')
        self.ds.load_data(split='val')
        self.dataset['train'] = self.ds.dataset['train']
        self.dataset['val'] = self.ds.dataset['val']
        callbacks = self.callbacks_list()
        opt = Adam(learning_rate=self.LEARNING_RATE)
        self.model.compile(loss=self.loss_func,
                           loss_weights=self.loss_weights,
                           optimizer=opt)
        self.model.fit(self.dataset['train'],
                       batch_size=self.BATCH_SIZE,
                       epochs=EPOCHS,
                       validation_data=self.dataset['val'],
                       callbacks=callbacks,
                       shuffle=True)
        wandb.log({"loss": self.loss_func})
        output_dir = os.path.join(self.SAVED_MODEL_PATH, f'{self.model_name}_{str(self.run_name)}')
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
            print(output_dir)
        self.model.save(output_dir)

    def resume_train_model(self, model_path):

        self.ds.load_data(split='train')
        self.ds.load_data(split='val')
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

        self.model.fit(self.dataset['train'],
                       batch_size=self.BATCH_SIZE,
                       epochs=EPOCHS,
                       validation_data=self.dataset['val'],
                       callbacks=callbacks,
                       shuffle=True)

    def pred_model(self, split):

        self.ds.load_data(split=split)
        self.dataset[split] = self.ds.dataset[split]
        if self.multi_tasking_flag:
            self.align[split], self.pred[split] = self.model.predict(self.dataset[split])
        else:
            self.pred[split] = self.model.predict(self.dataset[split])

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
            avg_pesq, avg_stoi, avg_psnr, avg_ssim, avg_mse = self.ds.spec2audio(split=split,
                                                                                 dec_pred=np.array(self.pred[split]),
                                                                                 algn_pred=None,
                                                                                 output_dir=storage_path,
                                                                                 sample_rate=self.sample_rate,
                                                                                 save_audio=True,
                                                                                 model_name=self.model_name,
                                                                                 informed=True)
        else:
            avg_pesq, avg_stoi, avg_psnr, avg_ssim, avg_mse = self.ds.spec2audio(split=split,
                                                                                 dec_pred=np.array(self.pred[split]),
                                                                                 algn_pred=np.array(self.align[split]),
                                                                                 output_dir=storage_path,
                                                                                 sample_rate=self.sample_rate,
                                                                                 save_audio=True,
                                                                                 model_name=self.model_name,
                                                                                 informed=True)

        return avg_pesq, avg_stoi, avg_psnr, avg_ssim, avg_mse

    def save_results(self, split):

        storage_path = os.path.join(self.SPEC_RESULTS_PATH, f'{self.model_name}_{split}_{self.run_name}')
        if not os.path.isdir(storage_path):
            os.makedirs(storage_path)
            print("created folder : ", storage_path)

        with open(storage_path + '/' + f'pred_{self.model_name}_{split}.pkl', 'wb') as result_file:
            pickle.dump(np.array(self.pred[split]), result_file)

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
        plt.title(f"Ground Truth of {split}")
        plt.imshow(self.data[split]["decoder_target"][idx, :, :])

        plt.show()
