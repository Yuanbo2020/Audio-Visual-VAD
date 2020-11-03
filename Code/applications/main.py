# coding= utf-8
import sys, os
sys.path.append(os.path.split(os.path.dirname(os.path.realpath(__file__)))[0])
import numpy as np
import framework.config as config
from framework.training_multi_loss import train_keras_generator_read_by_each_picture

np.random.seed(config.random_seed)


def main(argv):
    """
    Here it is decided whether the training is bimodal or single modality.
    """
    audioflag = 1 # 1 means there is this modal information input during training.
    imageflag = 1 # 1 means there is this modal information input during training.

    label_dependent = 0  # 0 means that the outputs of each branch are independent of each other.

    if audioflag and imageflag:
        information_type = ''
    elif audioflag:
        information_type = '_audio'
    elif imageflag:
        information_type = '_video'
    model_suffix = '_rule_embedded_' + str(regularize_frame_num) + 'frames_b' + str(config.batch) + '_e' + \
                   str(config.epoch) + '_m' + str(model_num) + information_type


    model_dir = os.path.join(config.output_model_system_path, 'model' + model_suffix)

    existed_training_agg_fps_path = os.path.join(config.existed_data_system_path, 'aggregator_reg2imageblock_'
                                                 + str(regularize_frame_num) + 'frames' + config.sampling_suffix)

    if not config.testing_mode:
        train_keras_generator_read_by_each_picture(model_num, model_dir, imageflag, audioflag,
                                                   existed_training_agg_fps_path, label_dependent,
                                                   config.video_input_frames_num, self_attention)


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except (ValueError, IOError) as e:
        sys.exit(e)

