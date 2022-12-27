import random
import pandas as pd
import os
import csv
import librosa
import numpy as np
from scipy.io.wavfile import read, write
from PSNR import PSNR
from pystoi import stoi
from pesq import pesq
import datetime
from imageio import imread
from skimage.transform import resize
import pickle
import tensorflow.keras.backend as K
from LipNet2.predict import predict
import shutil
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, MaxPool1D
from tensorflow.keras.applications import InceptionV3, inception_v3

class Dataset():

    def __init__(self, lipnet_model_path, multimodal_flag, lipnet_flag):

        self.CURRENT_PATH = '/Users/kadkhodm/Desktop/Research/inpainting/'
        self.run_name = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        self.df = {"train": pd.read_csv("train_data.csv"),
                   "test": pd.read_csv("test_data.csv")}
        self.lipnet_model_path = lipnet_model_path
        self.multimodal_flag = multimodal_flag
        self.lipnet_flag = lipnet_flag
        self.data = {}
        self.load_data('train')
        self.load_data('test')

    def build_data(self, split):

        results = self.facial_landmark_detection(df=self.df[split],
                                                 split=split,
                                                 model_path=self.lipnet_model_path,
                                                 gap_ratio_min=0.08, gap_ratio_max=0.32,
                                                 multimodal_flag=self.multimodal_flag,
                                                 lipnet_flag =self.lipnet_flag)
        return results

    def get_cache_path(self, split):
        return os.path.join(self.CURRENT_PATH,
                            'datasets',
                            f'cache_{split}').rstrip('/') + '.cache'

    def load_data(self, split):
        print("cache path", self.get_cache_path(split))
        if os.path.isfile(self.get_cache_path(split)):
            print(f"\nLoading {split} dataset from cache...")
            with open(self.get_cache_path(split), 'rb') as fp:
                self.data[split] = pickle.load(fp)
            print(f"{split} list", type(self.data[split]))
        else:
            print(f"\nEnumerating {split} dataset from disk...")
            # print("train path", self.train_path )
            self.data[split] = self.build_data(split)
            # print("train list", self.train_list)
            with open(self.get_cache_path(split), 'wb') as fp:
                pickle.dump((self.data[split]), fp)

    def mel_spec(self, wav, fs=16000):

        stft = librosa.stft(wav.astype("float16"), n_fft=512, hop_length=128)
        stft_mag, _ = librosa.magphase(stft)
        mel_scale = librosa.feature.melspectrogram(S=stft_mag, sr=fs)
        mel_sgram = librosa.amplitude_to_db(mel_scale, ref=np.min)

        return mel_sgram

    def mel_spec_inv(self, mel_sgram, fs=16000):

        mel_sgram_inv = librosa.db_to_amplitude(np.transpose(mel_sgram))
        S = librosa.feature.inverse.mel_to_stft(mel_sgram_inv, n_fft=512, sr=fs)
        y = librosa.griffinlim(S, n_fft=512, hop_length=128, n_iter=2)

        return y

    def harmonic_percussive(self, y):

        D = librosa.stft(y)
        D_harmonic, D_percussive = librosa.decompose.hpss(D, margin=8)
        percussive = librosa.istft(D_percussive)

        return percussive

    def noise_cancellation(self, y, sr=16000):

        S_full, phase = librosa.magphase(librosa.stft(y))
        S_filter = librosa.decompose.nn_filter(S_full,
                                               aggregate=np.median,
                                               metric='cosine',
                                               width=int(librosa.time_to_frames(2, sr=sr)))
        S_filter = np.minimum(S_full, S_filter)
        margin_i, margin_v = 2, 10
        power = 2

        mask_i = librosa.util.softmask(S_filter,
                                       margin_i * (S_full - S_filter),
                                       power=power)

        mask_v = librosa.util.softmask(S_full - S_filter,
                                       margin_v * S_filter,
                                       power=power)

        S_foreground = mask_v * S_full
        S_background = mask_i * S_full

        combined = np.multiply(S_foreground, np.exp(1j * phase))
        y_foreground = librosa.istft(combined)
        # wavfile.write('foreground.wav', 16000, y_foreground)

        combined = np.multiply(S_background, np.exp(1j * phase))
        y_background = librosa.istft(combined)
        # wavfile.write('background.wav', 16000, y_background)

        return y_foreground

    def padded_wav(self, wav, new_len):
        new_wav = np.zeros(new_len)
        new_wav[0:wav.shape[0]] = wav

        return new_wav

    def spec2audio(self, split, input, truth, results, name, output_dir, sample_rate=16000):

        if not os.path.isdir(os.path.join(output_dir)):
            os.makedirs(os.path.join(output_dir))
            print("created folder : ", os.path.join(output_dir))
        avg_stoi = 0.0
        avg_pesq = 0.0
        for cnt in range(len(results)):

            fs, wav = read(os.path.join(
                self.CURRENT_PATH, f'datasets/audio_{split}',
                name[cnt] + '.wav'))

            inverted_truth = wav  # mel_spec_inv(truth[cnt, :, :])

            inverted_input = self.noise_cancellation(self.mel_spec_inv(input[cnt, :, :]))
            inverted_padded_input = self.padded_wav(inverted_input, wav.shape[0])
            mask = np.nonzero(inverted_padded_input == 0)
            inverted_true_input = inverted_truth.copy()
            inverted_true_input[mask] = 0  # inverted_padded_input * 300

            inverted_mel_audio = self.mel_spec_inv(results[cnt, :, :])
            inverted_padded_mel_audio = self.padded_wav(inverted_mel_audio, wav.shape[0])
            restrd = inverted_truth.copy()
            restrd[mask] = inverted_padded_mel_audio[mask] * 300

            print(PSNR(results[cnt, :, :], truth[cnt, :, :]))
            print(stoi(inverted_truth, restrd, sample_rate, extended=False))
            print(pesq(sample_rate, inverted_truth, restrd, 'wb'))

            avg_stoi = avg_stoi + stoi(inverted_truth, restrd, sample_rate, extended=False)
            avg_pesq = avg_pesq + pesq(sample_rate, inverted_truth, restrd, 'wb')

            if not os.path.isdir(os.path.join(output_dir, str(name[cnt]))):
                os.makedirs(os.path.join(output_dir, str(name[cnt])))
                print("created folder : ", os.path.join(output_dir, str(name[cnt])))

            write(os.path.join(output_dir, str(name[cnt]),
                                       'input' + '.wav'), sample_rate,
                          inverted_true_input)
            write(os.path.join(output_dir, str(name[cnt]),
                                       'ground_truth' + '.wav'), sample_rate,
                          inverted_truth)
            write(os.path.join(output_dir, str(name[cnt]),
                                       'results' + '.wav'), sample_rate,
                          restrd)

            print(PSNR(results[cnt, :, :], truth[cnt, :, :]))
            print(stoi(inverted_truth, restrd, sample_rate, extended=False))
            print(pesq(sample_rate, inverted_truth, restrd, 'wb'))

        print("average pesq:", avg_pesq / cnt)
        print("average stoi:", avg_stoi / cnt)

        return avg_pesq / cnt, avg_stoi / cnt


    def mask_gen(self, w, h, gap_ratio_min=0.08, gap_ratio_max=0.32):

        mask = np.ones((h, w))
        gap_len_min = int(gap_ratio_min * w)
        gap_len_max = int(gap_ratio_max * w)
        gap_num = random.randint(1, 8)
        total_gap_len = random.randint(gap_len_min, gap_len_max)
        while (gap_num > 0) and (total_gap_len > 0):
            center = random.randint(1, w)
            gap = random.randint(1, total_gap_len)
            min_gap_time = np.max([center - (gap // 2), 0])
            max_gap_time = np.min([(center + (gap // 2)), w])
            mask[:, min_gap_time:max_gap_time] = 0
            gap_num += -1
            total_gap_len = total_gap_len - gap
            if total_gap_len < 2:
                break

        return mask

    def set_data(self, frames):
        data_frames = []
        for frame in frames:
            frame = frame.swapaxes(0, 1)  # swap width and height to form format W x H x C
            if len(frame.shape) < 3:
                frame = np.array([frame]).swapaxes(0, 2).swapaxes(0, 1)  # Add grayscale channel
            data_frames.append(frame)
        frames_n = len(data_frames)
        data_frames = np.array(data_frames)  # T x W x H x C
        if K.image_data_format() == 'channels_first':
            data_frames = np.rollaxis(data_frames, 3)  # C x T x W x H
        return data_frames

    def InceptionV3_feature_extractor(self):

        feature_extractor = InceptionV3(weights="imagenet", include_top=False,
                                        pooling="avg", input_shape=(75, 75, 3),)
        preprocess_input = inception_v3.preprocess_input
        inputs = Input((75, 75, 3))
        preprocessed = preprocess_input(inputs)
        imagenet_outputs = feature_extractor(preprocessed)
        resh_out = tf.expand_dims(imagenet_outputs, axis=-1)
        out_max = MaxPool1D(pool_size=64, strides=64, name='max')(resh_out)
        out = tf.squeeze(out_max)
        # dense = Dense(NUM_ENC_FEAT, activation='relu')
        # outputs = dense(imagenet_outputs)

        return Model(inputs, out, name="feature_extractor")

    def facial_landmark_detection(self, df, split, model_path,
                                  gap_ratio_min=0.08, gap_ratio_max=0.32,
                                  multimodal_flag=True, lipnet_flag=True):

        video_paths = df["video_path"].values.tolist()
        frame_features = []
        spec_mask = []
        spec_features = []
        spec_label = []
        names = []
        if not lipnet_flag:
            feature_extractor = self.InceptionV3_feature_extractor()
        for idx, path in enumerate(video_paths):  # enumerate(random.sample(video_paths, 1000)): #

            names.append(path.split(f"{split}/")[1])
            fs, wav = read(os.path.join(
                path.split(f"{split}/")[0] + f'audio_{split}',
                path.split(f"{split}/")[-1] + '.wav'))
            mel_sgram = self.mel_spec(wav, fs)
            h = mel_sgram.shape[0]
            w = mel_sgram.shape[1]
            mask2d = self.mask_gen(w, h, gap_ratio_min=gap_ratio_min, gap_ratio_max=gap_ratio_max)
            masked_mel_sgram = mel_sgram * mask2d

            spec_features.append(np.transpose(masked_mel_sgram))
            spec_label.append(np.transpose(mel_sgram))
            spec_mask.append(np.transpose(mask2d))

            if multimodal_flag:
                frames_paths = sorted([os.path.join(path, x) for x in os.listdir(path)])

                if lipnet_flag:
                    frames = [imread(frame_path) for frame_path in frames_paths]
                    frames = np.array(frames)
                    pred = predict(model_path,
                                   self.set_data(frames),
                                   absolute_max_string_len=32,
                                   output_size=28)
                else:
                    frames = [resize(imread(frame_path), (75, 75)) for frame_path in frames_paths]
                    frames = np.array(frames)
                    pred = feature_extractor.predict(frames)

                frame_features.append(np.array(pred).squeeze())
            print(f"collecting {idx} {split} features and remaining {len(video_paths) - idx} from : {path}.")

            if idx % 500 == 0 and idx > 0:
                if multimodal_flag:
                    set = {"names": names,
                           "encoder_input": frame_features,
                           "decoder_input": spec_features,
                           "decoder_mask": spec_mask,
                           "decoder_target": spec_label}
                    dataset_filename = os.path.join(self.CURRENT_PATH, 'datasets',
                                                    f'cache_{split}').rstrip('/') + '.cache'
                else:
                    set = {"names": names,
                           "decoder_input": spec_features,
                           "decoder_mask": spec_mask,
                           "decoder_target": spec_label}

                    dataset_filename = os.path.join(self.CURRENT_PATH, 'datasets',
                                                    f'cache_audio_{split}').rstrip('/') + '.cache'
                if os.path.isfile(dataset_filename):
                    os.remove(dataset_filename)
                    print(f"removed {dataset_filename}")

                with open(dataset_filename, 'wb') as fp:
                    pickle.dump(set, fp)

                print(f"saved {len(spec_features)} number of features in the {split} dataset")
        """
        if os.path.isfile(dataset_filename):
            os.remove(dataset_filename)
            print(f"removed {dataset_filename}")

        with open(dataset_filename, 'wb') as fp:
            pickle.dump(set, fp)
        """

        print(f"saved {len(spec_features)} number of features in the {split} dataset")

        return set

    def data_frame(self):
        path = os.path.join(self.CURRENT_PATH, 'datasets')
        header = ['cnt', 'video_path', 'audio_path', 'align_path']
        with open('train_data.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(header)
            video_path = os.path.join(path, 'train')
            align_path = os.path.join(path, 'align')
            spec_path = os.path.join(path, 'audio_train')
            cnt = 0
            spkrs = os.listdir(video_path)
            for spkr in spkrs:  # ['s24']:
                if os.path.isdir(os.path.join(video_path, spkr)):
                    folders = os.listdir(os.path.join(video_path, spkr))
                    for folder in folders:
                        if os.path.isdir(os.path.join(video_path, spkr, folder)):
                            frames = os.listdir(os.path.join(video_path, spkr, folder))
                            data = [cnt,
                                    os.path.join(video_path, spkr, folder),
                                    os.path.join(spec_path, spkr, folder + '.wav'),
                                    os.path.join(align_path, folder + '.align')]
                            writer.writerow(data)
                            cnt = cnt + 1

            header = ['cnt', 'video_path', 'audio_path', 'align_path']
            with open('test_data.csv', 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(header)
                video_path = os.path.join(path, 'test')
                align_path = os.path.join(path, 'align')
                spec_path = os.path.join(path, 'audio_test')
                cnt = 0
                spkrs = os.listdir(video_path)
                for spkr in spkrs:  # ['s2', 's20']:
                    if os.path.isdir(os.path.join(video_path, spkr)):
                        folders = os.listdir(os.path.join(video_path, spkr))
                        for folder in folders:
                            if os.path.isdir(os.path.join(video_path, spkr, folder)):
                                frames = os.listdir(os.path.join(video_path, spkr, folder))
                                data = [cnt,
                                        os.path.join(video_path, spkr, folder),
                                        os.path.join(spec_path, spkr, folder + '.wav'),
                                        os.path.join(align_path, folder + '.align')]
                                writer.writerow(data)
                                cnt = cnt + 1

    def train_test_val_split(self):
        src_path = os.path.join(self.CURRENT_PATH, 'datasets_original')
        dest_path = os.path.join(self.CURRENT_PATH, 'datasets')

        src_video_path = os.path.join(src_path, 'video_mouth_crop')
        # align_path = os.path.join(src_path, 'align')
        src_audio_path = os.path.join(src_path, 'audio_16k')
        cnt = 0
        rndm_test = 255
        spkrs = os.listdir(src_video_path)
        for spkr in spkrs:
            if os.path.isdir(os.path.join(src_video_path, spkr)):
                folders = os.listdir(os.path.join(src_video_path, spkr))

                random_test_list = random.sample(range(0, len(folders)), 255)
                for random_test in random_test_list:
                    if os.path.isdir(os.path.join(src_video_path, spkr, folders[random_test])):
                        src = os.listdir(os.path.join(src_video_path, spkr, folders[random_test]))
                        src_audio = os.path.join(src_audio_path, spkr, folders[random_test] + '.wav')
                        trg_audio = os.path.join(dest_path, 'audio_test', spkr)
                        if not (os.path.isdir(trg_audio)):
                            os.makedirs(trg_audio)
                            print("created folder : ", trg_audio)
                        shutil.copy2(src_audio, trg_audio)
                        for file in src:
                            if file.endswith('.png'):
                                # copying the files to the destination directory
                                src_video = os.path.join(src_video_path, spkr, folders[random_test], file)
                                trg_video = os.path.join(dest_path, 'test', spkr, folders[random_test])
                                if not (os.path.isdir(trg_video)):
                                    os.makedirs(trg_video)
                                    print("created folder : ", trg_video)
                                shutil.copy2(src_video, trg_video)

                random_train_list = [i for i in range(0, len(folders)) if i not in random_test_list]
                for random_train in random_train_list:
                    if os.path.isdir(os.path.join(src_video_path, spkr, folders[random_train])):
                        src = os.listdir(os.path.join(src_video_path, spkr, folders[random_train]))

                        src_audio = os.path.join(src_audio_path, spkr, folders[random_train] + '.wav')
                        trg_audio = os.path.join(dest_path, 'audio_train', spkr)
                        if not (os.path.isdir(trg_audio)):
                            os.makedirs(trg_audio)
                            print("created folder : ", trg_audio)
                        shutil.copy2(src_audio, trg_audio)
                        for file in src:
                            if file.endswith('.png'):
                                # copying the files to the destination directory
                                src_video = os.path.join(src_video_path, spkr, folders[random_train], file)
                                trg_video = os.path.join(dest_path, 'train', spkr, folders[random_train])
                                if not (os.path.isdir(trg_video)):
                                    os.makedirs(trg_video)
                                    print("created folder : ", trg_video)
                                shutil.copy2(src_video, trg_video)

    """
        self.audio_set = []
        self.video_set = []
        self.combine_sets('train')
        self.combine_sets('test')

        def get_cache_path(self, split, modality):

            return os.path.join(self.CURRENT_PATH, 'datasets', f'{modality}_cache_{split}').rstrip('/') + '.cache'

        def load_data(self, split, modality):

            print("cache path", self.get_cache_path(split, modality))
            if os.path.isfile(self.get_cache_path(split, modality)):
                print(f"Loading {split} dataset from cache...")
                with open(self.get_cache_path(split, modality), 'rb') as fp:
                    set = pickle.load(fp)
                print(f"{split} list", type(set))
            else:
                print(f"Enumerating {split} dataset from disk...")
                # print("train path", self.train_path )
                set = self.build_data(split)
                # print("train list", self.train_list)
                with open(self.get_cache_path(split, modality), 'wb') as fp:
                    pickle.dump(set, fp)

            return set

        def combine_sets(self, split):

            self.audio_set = self.load_data(split, 'audio')
            self.video_set = self.load_data(split, 'video')
            j = 0
            for i in range(0, len(self.video_set["names"])):
                if self.audio_set["names"][j] == self.video_set["names"][i]:
                    self.data[split]["encoder_input"][j] = self.video_set["encoder_input"][i]
                    self.data[split]["names"][j] = self.video_set["names"][i]
                    self.data[split]["decoder_input"][j] = self.audio_set["decoder_input"][j]
                    self.data[split]["decoder_mask"][j] = self.audio_set["decoder_mask"][j]
                    self.data[split]["decoder_target"][j] = self.audio_set["decoder_target"][j]
                    j += 1
                else:
                    i += 1
            return self.data
        """







