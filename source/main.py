import time
from CNN_RNN import Seq2seq
import datetime

start_time = time.time()
run_name = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
################################################    AV_MTL_S2S   #######################################################
AV_MTL_S2S = Seq2seq(multi_tasking_flag=True, multimodal_flag=True, seq2seq_flag=True, landmark_flag=True,)
# either load a saved model or train the model
# AV_MTL_S2S .load_model('../logs/AV_MTL_S2S_checkpoint', split='test')
AV_MTL_S2S.train_model()
AV_MTL_S2S.pred_model('test')
AV_MTL_S2S.save_results('test')
AV_MTL_S2S.spec2audio('test')
####################################################    AV_S2S   #######################################################

AV_S2S = Seq2seq(multi_tasking_flag=False, multimodal_flag=True, seq2seq_flag=True, landmark_flag=True,)
# AV_S2S.load_model('../logs/AV_S2S_checkpoint', split='test')
AV_S2S.train_model()
AV_S2S.pred_model('test')
AV_S2S.spec2audio('test')
AV_S2S.save_results('test')

######################################################    AV_SI   ######################################################
AV_SI = Seq2seq(multi_tasking_flag=False, multimodal_flag=True, seq2seq_flag=False, landmark_flag=True,)
# AV_SI.load_model('../logs/AV_SI_checkpoint', split='test')
AV_SI.train_model()
AV_SI.pred_model('test')
AV_SI.save_results('test')
AV_SI.spec2audio('test')
#####################################################    A_SI   ########################################################

A_SI = Seq2seq(multi_tasking_flag=False, multimodal_flag=False, seq2seq_flag=False, landmark_flag=False,)
# A_SI.load_model('../logs/A_SI_checkpoint', split='test')
A_SI.train_model()
A_SI.pred_model('test')
A_SI.save_results('test')
A_SI.spec2audio('test')
########################################################################################################################
end_time = time.time()
print(f"elapsed time is {(end_time - start_time) / 3600}")
