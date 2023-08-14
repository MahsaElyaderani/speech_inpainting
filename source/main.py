import time
from CNN_RNN import Seq2seq
from LipNet.train import train
import datetime
# import tensorflow
from LipNet.predict import lipnet_predict
# import optuna

# tensorflow.config.experimental.set_visible_devices([], 'GPU')
# tf.debugging.set_log_device_placement(True)
start_time = time.time()
run_name = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

# train(run_name=run_name, start_epoch=0, stop_epoch=200, img_c=1, img_w=100,
#      img_h=50, frames_n=149, absolute_max_string_len=28, minibatch_size=32,
#      model_path='checkpoint_lipnet_2023_03_06_01_02_21')
# seq2seq_train(split='val')
# lipnet_predict(split='test', lipnet_path='checkpoint_lipnet_2023_03_07_11_53_09', start_epoch=18)

# study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler())
# study.optimize(AV_MTL_S2S.objective, n_trials=1)
# print("Best trial:")
# trial = study.best_trial
# print("  Value: ", trial.value)
# print("  Params: ")
# for key, value in trial.params.items():
#    print("    {}: {}".format(key, value))
# best_params = study.best_params
########################################################################################################################
A_SI = Seq2seq(
    lipnet_model_path='/Users/kadkhodm/Desktop/Research/inpainting/saved_models/lipnet_model_2023_02_08_13_44_05',
    multi_tasking_flag=False, multimodal_flag=False, seq2seq_flag=False,
    attention_flag=False, lipnet_flag=False, landmark_flag=False,
    cnn_flag=False, vgg_flag=False, mtrc_flag=False)
A_SI.load_model('/Users/kadkhodm/Desktop/Research/inpainting/logs/A_SI_checkpoint',
                split='test_xl')
A_SI.load_model('/Users/kadkhodm/Desktop/Research/inpainting/logs/A_SI_checkpoint',
                split='test_l')
A_SI.load_model('/Users/kadkhodm/Desktop/Research/inpainting/logs/A_SI_checkpoint',
                split='test_s')
A_SI.load_model('/Users/kadkhodm/Desktop/Research/inpainting/logs/A_SI_checkpoint',
                split='test')
A_SI.load_model('/Users/kadkhodm/Desktop/Research/inpainting/logs/A_SI_checkpoint',
                split='val')
A_SI.load_model('/Users/kadkhodm/Desktop/Research/inpainting/logs/AUG_A_SI_checkpoint',
                split='test_env')
A_SI.load_model('/Users/kadkhodm/Desktop/Research/inpainting/logs/AUG_A_SI_checkpoint',
                split='test_noise')
A_SI.train_model()
A_SI.pred_model('test')
A_SI.save_results('test')
A_SI.spec2audio('test')
########################################################################################################################
AV_SI = Seq2seq(
    lipnet_model_path='/Users/kadkhodm/Desktop/Research/inpainting/saved_models/lipnet_model_2023_02_08_13_44_05',
    multi_tasking_flag=False, multimodal_flag=True, seq2seq_flag=False,
    attention_flag=False, lipnet_flag=False, landmark_flag=True,
    cnn_flag=False, vgg_flag=False, mtrc_flag=False)
AV_SI.load_model('/Users/kadkhodm/Desktop/Research/inpainting/logs/AV_SI_checkpoint',
                 split='test_xl')
AV_SI.load_model('/Users/kadkhodm/Desktop/Research/inpainting/logs/AV_SI_checkpoint',
                 split='test_l')
AV_SI.load_model('/Users/kadkhodm/Desktop/Research/inpainting/logs/AV_SI_checkpoint',
                 split='test_s')
AV_SI.load_model('/Users/kadkhodm/Desktop/Research/inpainting/logs/AV_SI_checkpoint',
                 split='test')
AV_SI.load_model('/Users/kadkhodm/Desktop/Research/inpainting/logs/AV_SI_checkpoint',
                 split='val')
AV_SI.load_model('/Users/kadkhodm/Desktop/Research/inpainting/logs/AV_SI_checkpoint',
                 split='test_env')
AV_SI.load_model('/Users/kadkhodm/Desktop/Research/inpainting/logs/AV_SI_checkpoint',
                 split='test_noise')
AV_SI.train_model()
AV_SI.pred_model('test')
AV_SI.save_results('test')
AV_SI.spec2audio('test')
########################################################################################################################
AV_S2S = Seq2seq(
    lipnet_model_path='/Users/kadkhodm/Desktop/Research/inpainting/saved_models/lipnet_model_2023_02_08_13_44_05',
    multi_tasking_flag=False, multimodal_flag=True, seq2seq_flag=True,
    attention_flag=False, lipnet_flag=False, landmark_flag=True,
    cnn_flag=False, vgg_flag=False, mtrc_flag=False)
AV_S2S.load_model('/Users/kadkhodm/Desktop/Research/inpainting/logs/AV_S2S_checkpoint_2023_03_23_19_36_31',
                  split='test_xl')
AV_S2S.load_model('/Users/kadkhodm/Desktop/Research/inpainting/logs/AV_S2S_checkpoint_2023_03_23_19_36_31',
                  split='test_l')
AV_S2S.load_model('/Users/kadkhodm/Desktop/Research/inpainting/logs/AV_S2S_checkpoint_2023_03_23_19_36_31',
                  split='test_s')
AV_S2S.load_model('/Users/kadkhodm/Desktop/Research/inpainting/logs/AV_S2S_checkpoint',
                  split='test')
AV_S2S.load_model('/Users/kadkhodm/Desktop/Research/inpainting/logs/AV_S2S_checkpoint_2023_03_23_19_36_31',
                  split='val')
AV_S2S.load_model('/Users/kadkhodm/Desktop/Research/inpainting/logs/AV_S2S_checkpoint',
                   split='test_env')
AV_S2S.load_model('/Users/kadkhodm/Desktop/Research/inpainting/logs/AV_S2S_checkpoint',
                  split='test_noise')
AV_S2S.train_model()
AV_S2S.pred_model('test')
AV_S2S.spec2audio('test')
AV_S2S.save_results('test')

########################################################################################################################
AV_MTL_SI = Seq2seq(
    lipnet_model_path='/Users/kadkhodm/Desktop/Research/inpainting/saved_models/lipnet_model_2023_02_08_13_44_05',
    multi_tasking_flag=True, multimodal_flag=True, seq2seq_flag=False,
    attention_flag=False, lipnet_flag=False, landmark_flag=True,
    cnn_flag=False, vgg_flag=False, mtrc_flag=False)
AV_MTL_SI.load_model('/Users/kadkhodm/Desktop/Research/inpainting/logs/AV_MTL_SI_checkpoint_2023_03_24_01_45_44',
                     split='test_xl')
AV_MTL_SI.load_model('/Users/kadkhodm/Desktop/Research/inpainting/logs/AV_MTL_SI_checkpoint_2023_03_24_01_45_44',
                     split='test_l')
AV_MTL_SI.load_model('/Users/kadkhodm/Desktop/Research/inpainting/logs/AV_MTL_SI_checkpoint_2023_03_24_01_45_44',
                     split='test_s')
AV_MTL_SI.load_model('/Users/kadkhodm/Desktop/Research/inpainting/logs/AV_MTL_SI_checkpoint_2023_03_24_01_45_44',
                     split='test')
AV_MTL_SI.load_model('/Users/kadkhodm/Desktop/Research/inpainting/logs/AV_MTL_SI_checkpoint_2023_03_24_01_45_44',
                     split='val')
AV_MTL_SI.load_model('/Users/kadkhodm/Desktop/Research/inpainting/logs/AV_MTL_SI_checkpoint',
                      split='test_env')
AV_MTL_SI.load_model('/Users/kadkhodm/Desktop/Research/inpainting/logs/AV_MTL_SI_checkpoint',
                      split='test_noise')
AV_MTL_SI.train_model()
AV_MTL_SI.resume_train_model('/Users/kadkhodm/Desktop/Research/inpainting/logs/checkpoint_AV_MTL_SI_landmarks_mse_2023_07_05_08_29_14')
AV_MTL_SI.pred_model('test')
AV_MTL_SI.save_results('test')
AV_MTL_SI.spec2audio('test')
########################################################################################################################
AV_MTL_S2S = Seq2seq(
    lipnet_model_path='/Users/kadkhodm/Desktop/Research/inpainting/saved_models/lipnet_model_2023_02_08_13_44_05',
    multi_tasking_flag=True, multimodal_flag=True, seq2seq_flag=True,
    attention_flag=False, lipnet_flag=False, landmark_flag=True,
    cnn_flag=False, vgg_flag=False, mtrc_flag=False)

# AV_MTL_S2S.resume_train_model('/Users/kadkhodm/Downloads/model-best.h5')
AV_MTL_S2S.load_model('/Users/kadkhodm/Desktop/Research/inpainting/logs/AV_MTL_S2S_checkpoint',
                      split='test_xl')
AV_MTL_S2S.load_model('/Users/kadkhodm/Desktop/Research/inpainting/logs/AV_MTL_S2S_checkpoint',
                      split='test_l')
AV_MTL_S2S.load_model('/Users/kadkhodm/Desktop/Research/inpainting/logs/AV_MTL_S2S_checkpoint',
                      split='test_s')
AV_MTL_S2S.load_model('/Users/kadkhodm/Desktop/Research/inpainting/logs/AV_MTL_S2S_checkpoint',
                      split='test')
AV_MTL_S2S.load_model('/Users/kadkhodm/Desktop/Research/inpainting/logs/AV_MTL_S2S_checkpoint',
                      split='val')
AV_MTL_S2S.load_model('/Users/kadkhodm/Desktop/Research/inpainting/logs/AUG_AV_MTL_S2S_checkpoint',
                      split='test_env')
AV_MTL_S2S.load_model('/Users/kadkhodm/Desktop/Research/inpainting/logs/AUG_AV_MTL_S2S_checkpoint',
                      split='test_noise')

AV_MTL_S2S.train_model()
AV_MTL_S2S.pred_model('test')
AV_MTL_S2S.save_results('test')
AV_MTL_S2S.spec2audio('test')

end_time = time.time()
print(f"elapsed time is {(end_time - start_time) / 3600}")
