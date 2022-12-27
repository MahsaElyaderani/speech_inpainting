from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard, CSVLogger, ModelCheckpoint
from LipNet2.generators import BasicGenerator
from LipNet2.callbacks import Statistics, Visualize
from LipNet2.curriculums import Curriculum
from LipNet2.decoders import Decoder
from LipNet2.helpers import labels_to_text
from LipNet2.spell import Spell
from LipNet2.model2 import LipNet
import numpy as np
import datetime
import os
import tensorflow.keras.losses as losses

np.random.seed(55)

CURRENT_PATH = '/Users/kadkhodm/Desktop/Research/inpainting/' #os.path.dirname(os.path.abspath('/Users/kadkhodm/Desktop/Research/Mitacs/inpainting_data/'))
DATASET_DIR  = os.path.join(CURRENT_PATH, 'datasets')
OUTPUT_DIR   = os.path.join(CURRENT_PATH, 'saved_models')
LOG_DIR      = os.path.join(CURRENT_PATH, 'logs')

PREDICT_GREEDY      = False
PREDICT_BEAM_WIDTH  = 200
PREDICT_DICTIONARY  = os.path.join(CURRENT_PATH, 'dictionaries', 'grid.txt')


def curriculum_rules(epoch):

    return { 'sentence_length': -1, 'flip_probability': 0.5, 'jitter_probability': 0.05 }


def train(run_name, start_epoch, stop_epoch, img_c, img_w, img_h, frames_n, absolute_max_string_len, minibatch_size):
    curriculum = Curriculum(curriculum_rules)
    lip_gen = BasicGenerator(dataset_path=DATASET_DIR,
                             minibatch_size=minibatch_size,
                             img_c=img_c, img_w=img_w, img_h=img_h, frames_n=frames_n,
                             absolute_max_string_len=absolute_max_string_len,
                             curriculum=curriculum, start_epoch=start_epoch).build()
    # print("lipgen type", type(lip_gen))

    lipnet = LipNet(img_c=img_c, img_w=img_w, img_h=img_h, frames_n=frames_n,
                    absolute_max_string_len=absolute_max_string_len, output_size=lip_gen.get_output_size())
    lipnet.summary()
    lipnet.partial_model.summary()

    adam = Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    # the loss calc occurs elsewhere, so use a dummy lambda func for the loss
    lipnet.model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=adam)
    # MAHSA: added custom_loss to the compile
    # custom_loss = {'ctc': lambda y_true, y_pred: y_pred}
    # losses.custom_loss = custom_loss
    # lipnet.model.compile(loss=custom_loss, optimizer=adam)

    # load weight if necessary
    if start_epoch > 0:
        weight_file = os.path.join(OUTPUT_DIR, os.path.join(run_name, 'weights%02d.h5' % (start_epoch - 1)))
        lipnet.model.load_weights(weight_file)

    spell = Spell(path=PREDICT_DICTIONARY)
    decoder = Decoder(greedy=PREDICT_GREEDY, beam_width=PREDICT_BEAM_WIDTH,
                      postprocessors=[labels_to_text, spell.sentence])

    # define callbacks
    # statistics  = Statistics(lipnet, lip_gen.next_val(), decoder, 256, output_dir=os.path.join(OUTPUT_DIR, run_name))
    # visualize   = Visualize(os.path.join(OUTPUT_DIR, run_name),
    #                         lipnet, lip_gen.next_val(),
    #                         decoder, num_display_sentences=minibatch_size)
    tensorboard = TensorBoard(log_dir=os.path.join(LOG_DIR, f'lipnet_{run_name}'))
    csv_logger  = CSVLogger(os.path.join(LOG_DIR, "{}-{}.csv".format('training', f'lipnet_{run_name}')),
                            separator=',',
                            append=True)
    checkpoint  = ModelCheckpoint(os.path.join(LOG_DIR, f'checkpoint_lipnet_{run_name}', "model{epoch:02d}.h5"),
                                  monitor='val_loss',
                                  save_weights_only=False,
                                  mode='auto', save_freq=1) ## Mahsa: set save_freq wisely
    lip_net_callbacks = [lip_gen, tensorboard, checkpoint, csv_logger]
    # lip_net_callbacks = [checkpoint, statistics, visualize, lip_gen, tensorboard, csv_logger]
    # print(f"coparing weights of partial model before fit {np.sum(lipnet.partial_model.layers[22].get_weights()[0])}")
    # print(f"coparing weights of model before fit {np.sum(lipnet.model.layers[22].get_weights()[0])}")
    lipnet.model.fit(lip_gen.next_train(),
                     steps_per_epoch=lip_gen.default_training_steps,
                     epochs=stop_epoch,
                     validation_data=lip_gen.next_val(),
                     validation_steps=lip_gen.default_validation_steps,
                     callbacks=lip_net_callbacks,
                     initial_epoch=start_epoch,
                     verbose=1,
                     max_queue_size=5,
                     workers=1,
                     use_multiprocessing=False,
                     # pickle_safe=True
                     )
    print(f"coparing weights of partial model after fit {np.sum(lipnet.partial_model.layers[22].get_weights()[0])}")
    print(f"coparing weights of model after fit {np.sum(lipnet.model.layers[22].get_weights()[0])}")

    if not os.path.isdir(os.path.join(OUTPUT_DIR, 'lipnet_model_'+str(run_name))):
        os.makedirs(os.path.join(OUTPUT_DIR, 'lipnet_model_'+str(run_name)))
        print(os.path.join(OUTPUT_DIR, 'lipnet_model_'+str(run_name)))
    lipnet.partial_model.save(os.path.join(OUTPUT_DIR,'lipnet_model_'+str(run_name)))


#if __name__ == '__main__':
#    run_name = datetime.datetime.now().strftime('%Y:%m:%d:%H:%M:%S')
#    train(run_name=run_name, start_epoch=0, stop_epoch=10, img_c=3, img_w=100,
#          img_h=50, frames_n=75, absolute_max_string_len=32, minibatch_size=1)