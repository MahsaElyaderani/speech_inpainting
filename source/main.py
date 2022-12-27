import time
from CNN_RNN import Seq2seq
from LipNet2.train import train
import datetime

# tf.debugging.set_log_device_placement(True)
start_time = time.time()
run_name = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
#train(run_name=run_name, start_epoch=0, stop_epoch=4, img_c=3, img_w=100,
#      img_h=50, frames_n=75, absolute_max_string_len=32, minibatch_size=16)

seq2seq = Seq2seq(lipnet_model_path='/Users/kadkhodm/Desktop/Research/inpainting/saved_models/lipnet_model_2022_10_24_23_10_50',
                  multimodal_flag=False,
                  attention_flag=False,
                  seq2seq_flag=False,
                  lipnet_flag=False,
                  cnn_flag=False)
seq2seq.train_model()
seq2seq.pred_model('test')
seq2seq.spec2audio('test')
seq2seq.save_results('test')
end_time = time.time()
print(f"elapsed time is {(end_time - start_time)/3600}")



