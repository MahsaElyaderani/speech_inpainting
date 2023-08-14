import pandas as pd
import csv
import datetime
import os
import pickle
import shutil

import cv2
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K
from editdistance import eval as edit_eval
from jiwer import wer
from pesq import pesq
from pystoi import stoi
from scipy.signal.windows import hann as hanning
from skimage.metrics import structural_similarity as ssim
from skimage.transform import resize
from imageio import imread
from tensorflow.keras import Model
from tensorflow.keras.applications import InceptionV3, inception_v3
from tensorflow.keras.layers import Input, MaxPool1D
from tensorflow.keras.models import load_model

# from LipNet2.predict import predict
from LipNet2.model2 import LipNet
from PSNR import PSNR
from tools_audio import *
from tools_video import extract_face_landmarks, get_motion_vector, sync_audio_visual_features
from tools_alignment import load_alignments, decode_batch_predictions, num_to_char

MAX_DEC_SEQ_LEN = 149
NUM_DEC_FEAT = 64
FRAME_HEIGHT = 75
FRAME_WIDTH = 100
FRAME_CHANNEL = 3


class Dataset():

    def __init__(self, lipnet_model_path, multimodal_flag,
                 lipnet_flag, landmark_flag, multi_tasking_flag):

        self.CURRENT_PATH = '/Users/kadkhodm/Desktop/Research/inpainting/'
        self.run_name = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        if not os.path.isfile("train_data.csv"):
            self.data_frame('train')
            self.data_frame('val')
            self.data_frame('test')

        self.df = {"train": pd.read_csv("train_data.csv"),
                   "val": pd.read_csv("val_data.csv"),
                   "test": pd.read_csv("test_data.csv"),
                   "test_s": pd.read_csv("test_data.csv"),
                   "test_l": pd.read_csv("test_data.csv"),
                   "test_xl": pd.read_csv("test_data.csv"),
                   "test_env": pd.read_csv("test_data.csv"),
                   "test_noise": pd.read_csv("test_data.csv")}
        self.lipnet_model_path = lipnet_model_path
        self.multimodal_flag = multimodal_flag
        self.lipnet_flag = lipnet_flag
        self.landmark_flag = landmark_flag
        self.multi_tasking_flag = multi_tasking_flag
        self.BATCH_SIZE = 32
        self.sample_rate = 8000  # 16000
        self.absolute_max_string_len = 41
        self.encoder_min = -41.778523
        self.encoder_max = 51.342281
        self.data = {}
        self.dataset = {}
        # self.load_data('train', augmented=True)
        # self.load_data('val', augmented=False)
        # self.load_data('test', augmented=False)
        # self.load_data('test_s', augmented=False)
        # self.load_data('test_l', augmented=False)
        # self.load_data('test_xl', augmented=False)
        # self.load_data('test_env', augmented=False)
        # self.load_data('test_noise', augmented=False)

    def build_data(self, split):
        if split == 'test_s':
            env_sound_flag = False
            mean_gaps = 300
            std_gaps = 300
        elif split == 'test_l':
            env_sound_flag = False
            mean_gaps = 800
            std_gaps = 300
        elif split == 'test_xl':
            env_sound_flag = False
            mean_gaps = 1300
            std_gaps = 300
        elif split == 'test_env':
            env_sound_flag = True
            mean_gaps = 900
            std_gaps = 300
        elif split == 'test_noise':
            env_sound_flag = True
            mean_gaps = 900
            std_gaps = 300
        else:
            env_sound_flag = False
            mean_gaps = 900
            std_gaps = 300

        results = self.multimodal_feature_extraction(df=self.df[split],
                                                     split=split,
                                                     model_path=self.lipnet_model_path,
                                                     multimodal_flag=self.multimodal_flag,
                                                     landmark_flag=self.landmark_flag,
                                                     multi_tasking_flag=self.multi_tasking_flag,
                                                     lipnet_flag=self.lipnet_flag,
                                                     incepv3_flag=False,
                                                     env_sound_flag=env_sound_flag,
                                                     mean_gaps=mean_gaps,
                                                     std_gaps=std_gaps)

        return results

    def get_cache_path(self, split):
        return os.path.join(self.CURRENT_PATH,
                            'datasets',
                            f'cache_{split}').rstrip('/') + '.cache'

    def load_data(self, split, augmented):
        print("cache path", self.get_cache_path(split))
        if os.path.isfile(self.get_cache_path(split)):
            print(f"\nLoading {split} dataset from cache...")
            with open(self.get_cache_path(split), 'rb') as fp:
                self.data[split] = pickle.load(fp)

            # self.data[split]["encoder_input"] = ((np.array(self.data[split]["encoder_input"]) - self.encoder_min) /
            #                                      (self.encoder_max - self.encoder_min)).astype('float32')
            encoder_input = np.array(self.data[split]["encoder_input"]).astype('float32')
            decoder_input = np.array(self.data[split]["decoder_input"]).astype('float32')
            decoder_target = np.array(self.data[split]["decoder_target"]).astype('float32')
            align = np.array(self.data[split]["align"]).astype('int64')
            if self.multimodal_flag and self.multi_tasking_flag:
                original_dataset = tf.data.Dataset.from_tensor_slices(((encoder_input, decoder_input),
                                                                       (align, decoder_target)))

            elif self.multimodal_flag and not self.multi_tasking_flag:
                original_dataset = tf.data.Dataset.from_tensor_slices(((encoder_input, decoder_input),
                                                                       decoder_target))

            elif not self.multimodal_flag and not self.multi_tasking_flag:
                original_dataset = tf.data.Dataset.from_tensor_slices((decoder_input, decoder_target))

            if augmented:

                aug1_dataset = (original_dataset.cache()
                                                .shuffle(self.BATCH_SIZE, reshuffle_each_iteration=True)
                                                .map(self.augment_noise, num_parallel_calls=tf.data.AUTOTUNE)
                                                .prefetch(tf.data.AUTOTUNE))
                aug2_dataset = (original_dataset.cache()
                                                .shuffle(self.BATCH_SIZE, reshuffle_each_iteration=True)
                                                .map(self.augment_env, num_parallel_calls=tf.data.AUTOTUNE)
                                                .prefetch(tf.data.AUTOTUNE))
                org_dataset = (original_dataset.cache()
                                               .shuffle(self.BATCH_SIZE, reshuffle_each_iteration=True)
                                               .prefetch(tf.data.AUTOTUNE))
                dataset = org_dataset.concatenate(aug1_dataset)
                dataset = dataset.concatenate(aug2_dataset)
                self.dataset[split] = (dataset.cache()
                                              .shuffle(self.BATCH_SIZE * 100, reshuffle_each_iteration=True)
                                              .batch(self.BATCH_SIZE, drop_remainder=True)
                                              .prefetch(tf.data.AUTOTUNE))

            else:
                self.dataset[split] = (original_dataset.cache()
                                                       .batch(self.BATCH_SIZE, drop_remainder=True)
                                                       .prefetch(tf.data.AUTOTUNE))
        else:
            print(f"\nEnumerating {split} dataset from disk...")
            # print("train path", self.train_path )
            self.data[split] = self.build_data(split)
            # print("train list", self.train_list)
            with open(self.get_cache_path(split), 'wb') as fp:
                pickle.dump((self.data[split]), fp)

    def augment_env(self, x, y):
        aug_mel_sgram = load_environmental_sound()
        if len(x) == 2:
            aug_decoder_input = x[1] + 1 * np.transpose(aug_mel_sgram)
            x_p = (x[0], aug_decoder_input)
        else:
            aug_decoder_input = x[0] + 1 * np.transpose(aug_mel_sgram)
            x_p = aug_decoder_input
        return x_p, y

    def augment_noise(self, x, y):
        if len(x) == 2:
            aug_decoder_input = x[1] + 0.1 * np.random.normal(0, 1, (MAX_DEC_SEQ_LEN, NUM_DEC_FEAT))
            x_p = (x[0], aug_decoder_input)
        else:
            aug_decoder_input = x[0] + 0.1 * np.random.normal(0, 1, (MAX_DEC_SEQ_LEN, NUM_DEC_FEAT))
            x_p = aug_decoder_input
        return x_p, y

    def multimodal_feature_extraction(self, df, split, model_path, multi_tasking_flag,
                                      multimodal_flag, lipnet_flag,
                                      landmark_flag, incepv3_flag,
                                      env_sound_flag=False, mean_gaps=900, std_gaps=300):

        video_paths = df["video_path"].values.tolist()
        frame_features = []
        spec_masks = []
        spec_features = []
        spec_labels = []
        names = []
        aligns = []
        flag_face_detect = 1

        if lipnet_flag:
            frames_n, img_w, img_h, img_c = 75, 100, 50, 3
            lipnet = LipNet(img_c=img_c, img_w=img_w, img_h=img_h, frames_n=frames_n,
                            absolute_max_string_len=32, output_size=28)
            lipnet.partial_model = load_model(model_path, compile=False)
        elif incepv3_flag:
            feature_extractor = self.inceptionV3_feature_extractor()

        for idx, path in enumerate(video_paths):  # enumerate(random.sample(video_paths, 1001)):#
            # print("path: ", path)
            if multimodal_flag:
                frames_paths = sorted([os.path.join(path, x) for x in os.listdir(path)])

                if lipnet_flag:
                    print("inside lipnet")
                    frames = [imread(frame_path) for frame_path in frames_paths]
                    frames = np.array(frames)
                    pred = lipnet.predict(np.expand_dims(self.set_data(frames), axis=0))

                elif landmark_flag:
                    print("inside Landmark")
                    if 'test_' in split:
                        # print(path.split("datasets")[0] + 'datasets_original/video_normal/' + path.split("test/")[-1] + '.mpg')
                        face_land, _, flag_face_found = extract_face_landmarks(
                            video_filename=os.path.join(path.split("datasets")[0] +
                                                        'datasets_original/video_normal/' +
                                                        path.split(f"{'test'}/")[-1] + '.mpg'),
                            predictor_params='shape_predictor_68_face_landmarks.dat',
                            refresh_size=8)
                    else:
                        face_land, _, flag_face_found = extract_face_landmarks(
                            video_filename=os.path.join(path.split("datasets")[0] +
                                                        'datasets_original/video_normal',
                                                        path.split(f"{split}/")[-1] + '.mpg'),
                            predictor_params='shape_predictor_68_face_landmarks.dat',
                            refresh_size=8)
                    flag_face_detect = flag_face_found

                    if flag_face_detect:
                        video_features = sync_audio_visual_features(MAX_DEC_SEQ_LEN, face_land.reshape(75, -1),
                                                                    tot_frames=75, min_frames=70)
                        if video_features is None:
                            print(f'Skipped. Video features of {path.split(f"{split}/")[1]} is corrupted.')
                            continue
                        # Compute landmark motion vector
                        pred = get_motion_vector(video_features, delta=1)
                        # pred = get_motion_vector(face_land, delta=1)
                        # pred = np.array(pred)
                        # print("pred shape", pred.shape)
                elif incepv3_flag:
                    print("inside Incepv3")
                    frames = [resize(imread(frame_path), (FRAME_HEIGHT, FRAME_WIDTH)) for frame_path in frames_paths]
                    frames = np.array(frames)
                    pred = feature_extractor.predict(frames)
                else:
                    print("inside PCA")
                    # frames = [imread(frame_path) for frame_path in frames_paths]
                    # gray = rgb2gray(frames)
                    # edges = self.lip_edges(gray)
                    # pred = np.array(np.expand_dims(edges, axis=-1))
                    # print(np.shape(edges))
                    # edges_flatten = edges.reshape((75, 5000))
                    # pca = PCA(n_components=3)
                    # pred = np.array([pca.fit_transform(edge) for edge in edges])
                    # pred = np.array(np.expand_dims(pred, axis=-1))

                if flag_face_detect:
                    frame_features.append(np.array(pred).squeeze())

            if flag_face_detect:
                if 'test_' in split:
                    name = path.split("test/")[1]
                    wav = load_wav(os.path.join(path.split("test/")[0] + f'audio_test',
                                                path.split("test/")[-1] + '.wav'), sr=self.sample_rate)
                else:
                    name = path.split(f"{split}/")[1]
                    wav = load_wav(os.path.join(path.split(f"{split}/")[0] + f'audio_{split}',
                                            path.split(f"{split}/")[-1] + '.wav'), sr=self.sample_rate)
                mel_sgram = melspectrogram(wav).astype(np.float16)
                mask2d, true_mask_cov, n_intr = get_intrusions_mask(mel_sgram.shape[0], mel_sgram.shape[1],
                                                                    mean_gaps / (len(wav) / 8),
                                                                    std_gaps / (len(wav) / 8),
                                                                    8, min_intr_len=3)
                spec_labels.append(np.transpose(mel_sgram))
                masked_mel_sgram = mel_sgram.copy()
                if multi_tasking_flag:
                    label = load_alignments(
                        os.path.join(self.CURRENT_PATH, 'datasets/align', os.path.split(name)[-1] + '.align'))
                    padding = np.zeros((self.absolute_max_string_len - len(label)))
                    align = np.concatenate((np.array(label), padding), axis=0)
                    aligns.append(align)
                if env_sound_flag:
                    if 'noise' in split:
                        masked_mel_sgram += 0.1 * np.random.normal(0, 1, (NUM_DEC_FEAT, MAX_DEC_SEQ_LEN))
                    elif 'env' in split:
                        masked_mel_sgram += 1 * load_environmental_sound()
                names.append(name)
                spec_features.append(np.transpose(masked_mel_sgram) * mask2d)
                spec_masks.append(mask2d)

            print(f"collecting {idx} {split} features and remaining {len(video_paths) - idx} from : {path}.")
            if idx % 40 == 0 and idx > 0:
                set = {"names": names,
                       "encoder_input": frame_features,
                       "decoder_input": spec_features,
                       "decoder_mask": spec_masks,
                       "align": aligns,
                       "decoder_target": spec_labels}
                dataset_filename = os.path.join(self.CURRENT_PATH, 'datasets',
                                                f'cache_{split}').rstrip('/') + '.cache'
                if os.path.isfile(dataset_filename):
                    os.remove(dataset_filename)
                    print(f"removed {dataset_filename}")

                with open(dataset_filename, 'wb') as fp:
                    pickle.dump(set, fp)

                print(f"saved {len(spec_features)} number of features in the {split} dataset")
        if os.path.isfile(dataset_filename):
            os.remove(dataset_filename)
            print(f"removed {dataset_filename}")

        with open(dataset_filename, 'wb') as fp:
            pickle.dump(set, fp)

        print(f"saved {len(spec_features)} number of features in the {split} dataset")

        return set

    def spec2audio(self, split, output_dir, sample_rate,
                   dec_pred, model_name, algn_pred=None, informed=True, save_audio=True):

        with open(os.path.join(self.CURRENT_PATH, 'datasets', f'cache_{split}.cache'), 'rb') as f:
            data = pickle.load(f)
        input = np.array(data["decoder_input"])
        truth = np.array(data["decoder_target"])
        mask = np.array(data["decoder_mask"])
        algn_truth = np.array(data["align"])
        name = np.array(data["names"])
        if informed:
            mask_not = np.array([np.where(mask[x] == 0, 1, 0) for x in range(0, len(dec_pred))])
            results = dec_pred * mask_not[:len(dec_pred)] + input[:len(dec_pred)]
        else:
            results = dec_pred

        # set = {"names": name,
        #        "decoder_input": input,
        #        "decoder_mask": mask,
        #        "decoder_target": truth,
        #        "decoder_pred": results}
        # dataset_filename = os.path.join(self.CURRENT_PATH, 'spec_results',
        #                                 f'cache_{split}').rstrip('/') + '.cache'
        # with open(dataset_filename, 'wb') as fp:
        #     pickle.dump(set, fp)

        total = len(results)
        avg_stoi = 0.0
        avg_pesq = 0.0
        avg_psnr = 0.0
        avg_mse = 0.0
        avg_l1 = 0.0
        avg_ssim = 0.0
        avg_wer = 0.0
        avg_per = 0.0

        avg_stoi_inp = 0.0
        avg_pesq_inp = 0.0
        avg_psnr_inp = 0.0
        avg_mse_inp = 0.0
        avg_l1_inp = 0.0
        avg_ssim_inp = 0.0
        avg_wer_inp = 0.0

        pesq_mode = 'nb'
        if sample_rate == 16000:
            pesq_mode = 'wb'
        elif sample_rate == 8000:
            pesq_mode = 'nb'

        if not os.path.isdir(os.path.join(output_dir)):
            os.makedirs(os.path.join(output_dir))
            print("created folder : ", os.path.join(output_dir))

        for cnt in range(len(results)):

            print(cnt)
            # wav = load_wav(os.path.join(self.CURRENT_PATH, f'datasets/audio_{split}',
            #                            name[cnt] + '.wav'), sample_rate)

            inv_true_input = inv_melspectrogram(np.transpose(input[cnt, :, :]))
            # inv_truth = wav[:len(inv_true_input)]
            inv_truth = inv_melspectrogram(np.transpose(truth[cnt, :, :]))
            # inv_input = inv_true_input.copy()
            # inv_input[np.abs(inv_input) <= 0.001] = 0.0
            # mask = np.nonzero(inv_input == 0)
            # mask = np.nonzero(inv_input == 0)
            # inv_true_input = inv_truth.copy()
            # inv_true_input[mask] = 0
            audio_res = inv_melspectrogram(np.transpose(results[cnt, :, :]))
            # audio_res = inv_truth.copy()
            # audio_res[mask] = inv_res[mask]
            if algn_pred is not None:
                sen_pred = decode_batch_predictions(algn_pred[cnt, :, :])
                sen_tar = tf.strings.reduce_join(num_to_char(((algn_truth[cnt])))).numpy().decode('utf-8')
                avg_wer = avg_wer + wer(sen_tar, sen_pred)
                avg_per += edit_eval(sen_tar, sen_pred) / len(sen_tar)
                print("-" * 100)
                print(f"Target    : {sen_tar}")
                print(f"Prediction: {sen_pred}")
                # print("wer_score", wer_score)
            print("PSNR:", PSNR(results[cnt, :, :], truth[cnt, :, :]))
            print("MSE:", self.maskd_acc(truth[cnt, :, :], results[cnt, :, :], mask[cnt, :, :])[1])
            print("SSIM:", ssim(results[cnt, :, :], truth[cnt, :, :]))
            print("STOI:", stoi(inv_truth, audio_res, sample_rate, extended=False))
            print("PESQ:", pesq(sample_rate, inv_truth, audio_res, mode=pesq_mode))

            avg_stoi += stoi(inv_truth, audio_res, sample_rate, extended=False)
            avg_pesq += pesq(sample_rate, inv_truth, audio_res, mode=pesq_mode)
            avg_psnr += PSNR(results[cnt, :, :], truth[cnt, :, :])[0]
            avg_mse += self.maskd_acc(truth[cnt, :, :], results[cnt, :, :], mask[cnt, :, :])[1]
            avg_l1 += self.maskd_acc(truth[cnt, :, :], results[cnt, :, :], mask[cnt, :, :])[0]
            avg_ssim += ssim(results[cnt, :, :], truth[cnt, :, :])

            avg_stoi_inp += stoi(inv_truth, inv_true_input, sample_rate, extended=False)
            avg_pesq_inp += pesq(sample_rate, inv_truth, inv_true_input, mode=pesq_mode)
            avg_psnr_inp += PSNR(input[cnt, :, :], truth[cnt, :, :])[0]
            avg_mse_inp += self.maskd_acc(truth[cnt, :, :], input[cnt, :, :], mask[cnt, :, :])[1]
            avg_l1_inp += self.maskd_acc(truth[cnt, :, :], results[cnt, :, :], mask[cnt, :, :])[0]
            avg_ssim_inp += ssim(input[cnt, :, :], truth[cnt, :, :])

            if save_audio:
                if not os.path.isdir(os.path.join(output_dir, str(name[cnt]))):
                    os.makedirs(os.path.join(output_dir, str(name[cnt])))
                    print("created folder : ", os.path.join(output_dir, str(name[cnt])))

                save_wav(inv_true_input, os.path.join(output_dir, str(name[cnt]), 'input.wav'))
                save_wav(inv_truth, os.path.join(output_dir, str(name[cnt]), 'ground_truth.wav'))
                save_wav(audio_res, os.path.join(output_dir, str(name[cnt]), 'results.wav'))

        print("-" * 100)
        print("average wer:", avg_wer / total)
        print("average per:", avg_per / total)
        print("average pesq:", avg_pesq / total)
        print("average stoi:", avg_stoi / total)
        print("average psnr:", avg_psnr / total)
        print("average ssim:", avg_ssim / total)
        print("average mse:", avg_mse / total)
        print("average l1:", avg_l1 / total)

        print("-" * 100)
        print("average pesq_inp:", avg_pesq_inp / total)
        print("average stoi_inp:", avg_stoi_inp / total)
        print("average psnr_inp:", avg_psnr_inp / total)
        print("average ssim_inp:", avg_ssim_inp / total)
        print("average mse_inp:", avg_mse_inp / total)
        print("average l1_inp:", avg_l1_inp / total)

        header = ['wer', 'per', 'pesq', 'stoi', 'psnr', 'ssim', 'mse', 'l1',
                  'pesq_inp', 'stoi_inp', 'psnr_inp', 'ssim_inp', 'mse_inp', 'l1_inp']
        with open(f'{split}_data_{model_name}_{self.run_name}.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(header)
            writer.writerow([avg_wer / total, avg_per / total, avg_pesq / total, avg_stoi / total,
                             avg_psnr / total, avg_ssim / total, avg_mse / total, avg_l1 / total,
                             avg_pesq_inp / total, avg_stoi_inp / total, avg_psnr_inp / total,
                             avg_ssim_inp / total, avg_mse_inp / total, avg_l1_inp / total])

        return avg_pesq / total, avg_stoi / total, avg_psnr / total, avg_ssim / total, avg_mse / total

    def maskd_acc(self, truth, pred, mask):
        mask_not = np.array(np.where(mask == 0, 1, 0), dtype=np.float32)
        msk_ones = np.array(mask_not).sum()
        l2 = np.sum((np.square(truth - pred)) * mask_not) * 1.0 / (msk_ones * 1.0)
        l1 = np.sum((np.abs(truth - pred)) * mask_not) * 1.0 / (msk_ones * 1.0)

        return l1, l2

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

    def inceptionV3_feature_extractor(self):
        feature_extractor = InceptionV3(weights="imagenet", include_top=False,
                                        pooling="avg",
                                        input_shape=(FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNEL), )
        preprocess_input = inception_v3.preprocess_input
        inputs = Input((FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNEL))
        preprocessed = preprocess_input(inputs)
        imagenet_outputs = feature_extractor(preprocessed)
        resh_out = tf.expand_dims(imagenet_outputs, axis=-1)
        out_max = MaxPool1D(pool_size=32, strides=32, name='max')(resh_out)
        out = tf.squeeze(out_max)
        # dense = Dense(NUM_ENC_FEAT, activation='relu')
        # outputs = dense(imagenet_outputs)
        return Model(inputs, out, name="feature_extractor")

    def data_frame(self, split):
        path = os.path.join(self.CURRENT_PATH, 'datasets')
        header = ['cnt', 'video_path', 'audio_path', 'align_path']
        with open(f'{split}_data.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(header)
            video_path = os.path.join(path, f'{split}')
            align_path = os.path.join(path, 'align')
            spec_path = os.path.join(path, f'audio_{split}')
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

    def mask_gen(self, w, h, gap_ratio_min, gap_ratio_max):
        mask = np.ones((h, w))
        gap_len_min = int(gap_ratio_min * w)
        gap_len_max = int(gap_ratio_max * w)
        gap_num = random.randint(1, 8)
        total_gap_len = random.randint(gap_len_min, gap_len_max)
        while (gap_num > 0):
            center = random.randint(self.min_gap_cen, w - self.min_gap_cen)
            gap = random.randint(self.min_gap_len, self.max_gap_len)
            min_gap_time = np.max([center - (gap // 2), 0])
            max_gap_time = np.min([(center + (gap // 2)), w])
            mask[:, min_gap_time:max_gap_time] = 0
            gap_num += -1
            total_gap_len = total_gap_len - (max_gap_time - min_gap_time)  # MAHSA: changed total_gap_len-gap
            if total_gap_len < self.min_gap_len:
                break

        return mask

    def train_test_val_split(self):
        src_path = os.path.join(self.CURRENT_PATH, 'datasets_original')
        dest_path = os.path.join(self.CURRENT_PATH, 'datasets')

        src_video_path = os.path.join(src_path, 'video_mouth_crop')
        # align_path = os.path.join(src_path, 'align')
        src_audio_path = os.path.join(src_path, 'audio_8k')
        cnt = 0
        rndm_test = 500
        spkrs = os.listdir(src_video_path)
        for spkr in spkrs:  # ['s26', 's27', 's29', 's31']:#
            if os.path.isdir(os.path.join(src_video_path, spkr)):
                folders = os.listdir(os.path.join(src_video_path, spkr))

                random_test_list = random.sample(range(0, len(folders)), rndm_test)
                for random_test in random_test_list:
                    if os.path.isdir(os.path.join(src_video_path, spkr, folders[random_test])):
                        src = os.listdir(os.path.join(src_video_path, spkr, folders[random_test]))
                        src_audio = os.path.join(src_audio_path, spkr, folders[random_test] + '.wav')
                        trg_audio = os.path.join(dest_path, 'audio_test_seen', spkr)
                        if not (os.path.isdir(trg_audio)):
                            os.makedirs(trg_audio)
                            print("created folder : ", trg_audio)
                        shutil.copy2(src_audio, trg_audio)
                        for file in src:
                            if file.endswith('.png'):
                                # copying the files to the destination directory
                                src_video = os.path.join(src_video_path, spkr, folders[random_test], file)
                                trg_video = os.path.join(dest_path, 'test_seen', spkr, folders[random_test])
                                if not (os.path.isdir(trg_video)):
                                    os.makedirs(trg_video)
                                    print("created folder : ", trg_video)
                                shutil.copy2(src_video, trg_video)

                random_train_list = [i for i in range(0, len(folders)) if i not in random_test_list]
                for random_train in random_train_list:
                    if os.path.isdir(os.path.join(src_video_path, spkr, folders[random_train])):
                        src = os.listdir(os.path.join(src_video_path, spkr, folders[random_train]))

                        src_audio = os.path.join(src_audio_path, spkr, folders[random_train] + '.wav')
                        trg_audio = os.path.join(dest_path, 'audio_val', spkr)
                        if not (os.path.isdir(trg_audio)):
                            os.makedirs(trg_audio)
                            print("created folder : ", trg_audio)
                        shutil.copy2(src_audio, trg_audio)
                        for file in src:
                            if file.endswith('.png'):
                                # copying the files to the destination directory
                                src_video = os.path.join(src_video_path, spkr, folders[random_train], file)
                                trg_video = os.path.join(dest_path, 'val', spkr, folders[random_train])
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
