import csv
import datetime
import shutil
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K
from editdistance import eval as edit_eval
from jiwer import wer
from pesq import pesq
from pystoi import stoi
from skimage.metrics import structural_similarity as ssim
from PSNR import PSNR
from tools_audio import *
from tools_video import extract_face_landmarks, get_motion_vector, sync_audio_visual_features
from tools_alignment import load_alignments, decode_batch_predictions, num_to_char

MAX_DEC_SEQ_LEN = 149
NUM_DEC_FEAT = 64
FRAME_HEIGHT = 75
FRAME_WIDTH = 100
FRAME_CHANNEL = 3


class Dataset:

    def __init__(self, multimodal_flag, landmark_flag, multi_tasking_flag):

        self.CURRENT_PATH = '../'
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
        self.multimodal_flag = multimodal_flag
        self.landmark_flag = landmark_flag
        self.multi_tasking_flag = multi_tasking_flag
        self.BATCH_SIZE = 32
        self.sample_rate = 8000
        self.absolute_max_string_len = 41
        self.data = {}
        self.dataset = {}

    def build_data(self, split):
        mean_gaps = 900
        std_gaps = 300

        results = self.multimodal_feature_extraction(df=self.df[split],
                                                     split=split,
                                                     multimodal_flag=self.multimodal_flag,
                                                     landmark_flag=self.landmark_flag,
                                                     multi_tasking_flag=self.multi_tasking_flag,
                                                     mean_gaps=mean_gaps,
                                                     std_gaps=std_gaps)

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

            self.dataset[split] = (original_dataset.cache()
                                                   .batch(self.BATCH_SIZE, drop_remainder=True)
                                                   .prefetch(tf.data.AUTOTUNE))
        else:
            print(f"\nEnumerating {split} dataset from disk...")
            self.data[split] = self.build_data(split)
            with open(self.get_cache_path(split), 'wb') as fp:
                pickle.dump((self.data[split]), fp)

    def multimodal_feature_extraction(self, df, split, multi_tasking_flag,
                                      multimodal_flag, landmark_flag, mean_gaps=900, std_gaps=300):

        video_paths = df["video_path"].values.tolist()
        frame_features = []
        spec_masks = []
        spec_features = []
        spec_labels = []
        names = []
        aligns = []
        flag_face_detect = 1

        for idx, path in enumerate(video_paths):
            if multimodal_flag:
                frames_paths = sorted([os.path.join(path, x) for x in os.listdir(path)])

                if landmark_flag:
                    print("inside Landmark")
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

                if flag_face_detect:
                    frame_features.append(np.array(pred).squeeze())

            if flag_face_detect:
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
                        os.path.join(path.split("datasets")[0], 'datasets/align', os.path.split(name)[-1] + '.align'))
                    padding = np.zeros((self.absolute_max_string_len - len(label)))
                    align = np.concatenate((np.array(label), padding), axis=0)
                    aligns.append(align)

                names.append(name)
                spec_features.append(np.transpose(masked_mel_sgram) * mask2d)
                spec_masks.append(mask2d)

            print(f"collecting {idx} {split} features and remaining {len(video_paths) - idx} from : {path}.")
            if idx % 50 == 0 and idx > 0:
                set = {"names": names,
                       "encoder_input": frame_features,
                       "decoder_input": spec_features,
                       "decoder_mask": spec_masks,
                       "align": aligns,
                       "decoder_target": spec_labels}
                dataset_filename = os.path.join(self.CURRENT_PATH, 'datasets',
                                                f'cache_{split}').rstrip('/') + '.cache'
                if not dataset_filename:
                    # Create a new directory because it does not exist
                    os.makedirs(dataset_filename)
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
            inv_true_input = inv_melspectrogram(np.transpose(input[cnt, :, :]))
            inv_truth = inv_melspectrogram(np.transpose(truth[cnt, :, :]))
            audio_res = inv_melspectrogram(np.transpose(results[cnt, :, :]))
            if algn_pred is not None:
                sen_pred = decode_batch_predictions(algn_pred[cnt, :, :])
                sen_tar = tf.strings.reduce_join(num_to_char((algn_truth[cnt]))).numpy().decode('utf-8')
                avg_wer = avg_wer + wer(sen_tar, sen_pred)
                avg_per += edit_eval(sen_tar, sen_pred) / len(sen_tar)
                print("-" * 100)
                print(f"Target    : {sen_tar}")
                print(f"Prediction: {sen_pred}")
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
