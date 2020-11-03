# coding=utf-8
import os, socket

# 系统设置
hostname=socket.gethostname()
ip = socket.gethostbyname(hostname)
""" path """
if ip == '10.100.196.192':
    existed_data_path = '' #　your data path
    """ evaluation """
    reference_path = os.path.join('')  #　your data path
elif ip == '10.100.196.194':
    existed_data_path = ''  #　your data path
    """ evaluation """
    reference_path = os.path.join('')  #　your data path

existed_data_system_path = None #　your data path

random_seed = 1024
label_list = ['singing', 'speech', 'silence', 'others']

video_input_frames_num = 3

# 帧率设置
fps = 3
block_hop_len = 1 / fps
sampling_suffix = '_' + str(fps) + 'fps'
frame_down_size = int(45/fps)

post_frame = fps  # 3
event_minimum_duration = 0

# 训练参数设置
testing_mode = 0  # True  # False
single_testing = 0  # 0  # 是否只测试单个文件， 1：只测试一个文件
continue_training = 0  # 0  # 是否沿着已有的模型继续，1：继续

# 模型输入信息设置
image_box_height960_width544 = (450, 300)
image_normalization = 1

epoch = 1 if single_testing else 100
batch = 64


# 输入图像
saving_picture_id = 0  # 0 : jpg, 1: png
frame_block_picture_format = ['jpg', 'png']


# 输出目录设置
current_path = os.getcwd()
output_model_system_path = os.path.join(current_path, 'system' + sampling_suffix)


# 各种loss
loss_list_label_dependent_double_modal = ['total_loss', 'joint_loss', 'audio_loss', 'video_loss',
                         'val_total_loss', 'val_joint_loss', 'val_audio_loss', 'val_video_loss']
loss_list_label_dependent_audio = ['loss', 'val_loss']

loss_list_label_independent_double_modal_self_attention = ['total_loss',
                             'audio_singing_output_loss', 'audio_speech_output_loss',
                             'audio_silence_output_loss', 'audio_others_output_loss',
                             'video_final_out_loss',
                             'joint_singing_out_loss', 'joint_speech_out_loss',
                             'joint_silence_out_loss', 'joint_others_out_loss',
                             'val_total_loss',
                             'val_audio_singing_output_loss', 'val_audio_speech_output_loss',
                             'val_audio_silence_output_loss', 'val_audio_others_output_loss',
                             'val_video_final_out_loss',
                             'val_joint_singing_out_loss', 'val_joint_speech_out_loss',
                             'val_joint_silence_out_loss', 'val_joint_others_out_loss']
loss_list_label_independent_double_modal_rule_embedded = ['total_loss',
                             'audio_singing_output_loss', 'audio_speech_output_loss',
                             'audio_silence_output_loss', 'audio_others_output_loss',
                             'video_open_output_loss', 'video_close_output_loss',
                             'joint_singing_out_loss', 'joint_speech_out_loss',
                             'joint_silence_out_loss', 'joint_others_out_loss',
                             'val_total_loss',
                             'val_audio_singing_output_loss', 'val_audio_speech_output_loss',
                             'val_audio_silence_output_loss', 'val_audio_others_output_loss',
                             'val_video_open_output_loss', 'val_video_close_output_loss',
                             'val_joint_singing_out_loss', 'val_joint_speech_out_loss',
                             'val_joint_silence_out_loss', 'val_joint_others_out_loss']
loss_list_label_independent_audio = ['loss',
                         'audio_singing_output_loss', 'audio_speech_output_loss',
                         'audio_silence_output_loss', 'audio_others_output_loss',
                         'val_loss',
                         'val_audio_singing_output_loss', 'val_audio_speech_output_loss',
                         'val_audio_silence_output_loss', 'val_audio_others_output_loss']











