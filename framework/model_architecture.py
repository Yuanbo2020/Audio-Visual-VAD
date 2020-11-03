#coding= utf-8
import numpy as np
from keras.layers import Input, merge, Reshape, Dense
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape, Permute,Lambda, RepeatVector
from keras.layers.convolutional import ZeroPadding2D, AveragePooling2D, Conv2D,MaxPooling2D, Convolution1D, MaxPooling1D, AveragePooling1D
from keras.layers.pooling import GlobalMaxPooling2D, GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import *
from keras.layers import LSTM, SimpleRNN, GRU, TimeDistributed, Bidirectional
from keras.models import Model, load_model
from keras import layers
from keras.optimizers import Adam
import framework.config as config
from keras import backend as K

np.random.seed(config.random_seed)


def joint_independent_compile(model):
    model.compile(loss={
        'audio_singing_output': 'binary_crossentropy',
        'audio_speech_output': 'binary_crossentropy',
        'audio_silence_output': 'binary_crossentropy',
        'audio_others_output': 'binary_crossentropy',

        'video_open_output': 'binary_crossentropy',
        'video_close_output': 'binary_crossentropy',

        'joint_singing_out': 'binary_crossentropy',
        'joint_speech_out': 'binary_crossentropy',
        'joint_silence_out': 'binary_crossentropy',
        'joint_others_out': 'binary_crossentropy'},
        loss_weights={
            'audio_singing_output': 1.0,
            'audio_speech_output': 1.0,
            'audio_silence_output': 1.0,
            'audio_others_output': 1.0,

            'video_open_output': 1.0,
            'video_close_output': 1.0,

            'joint_singing_out': 1.0,
            'joint_speech_out': 1.0,
            'joint_silence_out': 1.0,
            'joint_others_out': 1.0},
        optimizer='adam', metrics=['accuracy'])


def joint_independent_compile_attention(model):
    model.compile(loss={
        'audio_singing_output': 'binary_crossentropy',
        'audio_speech_output': 'binary_crossentropy',
        'audio_silence_output': 'binary_crossentropy',
        'audio_others_output': 'binary_crossentropy',

        'video_final_out': 'binary_crossentropy',

        'joint_singing_out': 'binary_crossentropy',
        'joint_speech_out': 'binary_crossentropy',
        'joint_silence_out': 'binary_crossentropy',
        'joint_others_out': 'binary_crossentropy'},
        loss_weights={
            'audio_singing_output': 1.0,
            'audio_speech_output': 1.0,
            'audio_silence_output': 1.0,
            'audio_others_output': 1.0,

            'video_final_out': 1.0,

            'joint_singing_out': 1.0,
            'joint_speech_out': 1.0,
            'joint_silence_out': 1.0,
            'joint_others_out': 1.0},
        optimizer='adam', metrics=['accuracy'])


def audio_independent_compile(model):
    model.compile(loss={
        'audio_singing_output': 'binary_crossentropy',
        'audio_speech_output': 'binary_crossentropy',
        'audio_silence_output': 'binary_crossentropy',
        'audio_others_output': 'binary_crossentropy'},
        loss_weights={
            'audio_singing_output': 1.0,
            'audio_speech_output': 1.0,
            'audio_silence_output': 1.0,
            'audio_others_output': 1.0},
        optimizer='adam', metrics=['accuracy'])


def audio_pooling_layer(audio_c):
    audio_p = layers.Conv2D(32, (3, 3), strides=(1, 2), padding='same')(audio_c)  # (None, 15, 32, 32)
    audio_p = layers.BatchNormalization()(audio_p)
    audio_p = layers.ReLU()(audio_p)
    return audio_p


def audio_cnn_layer1(audio_input):
    audio_c1_l = layers.Conv2D(32, (3, 7), padding='same')(audio_input)  # (N, 15, 64, 32)
    audio_c1_l = layers.BatchNormalization()(audio_c1_l)
    audio_c1_l = Activation('linear')(audio_c1_l)

    audio_c1_s = layers.Conv2D(32, (3, 7), padding='same')(audio_input)  # (N, 15, 64, 32)
    audio_c1_s = layers.BatchNormalization()(audio_c1_s)
    audio_c1_s = Activation('sigmoid')(audio_c1_s)

    audio_c1 = Multiply()([audio_c1_l, audio_c1_s])
    audio_p1 = audio_pooling_layer(audio_c1)
    return audio_p1


def audio_cnn_layer2(audio_p1):
    audio_c2_l = layers.Conv2D(32, (3, 7), padding='same')(audio_p1)  # (N, 15, 32, 32)
    audio_c2_l = layers.BatchNormalization()(audio_c2_l)
    audio_c2_l = Activation('linear')(audio_c2_l)

    audio_c2_s = layers.Conv2D(32, (3, 7), padding='same')(audio_p1)  # (N, 15, 32, 32)
    audio_c2_s = layers.BatchNormalization()(audio_c2_s)
    audio_c2_s = Activation('sigmoid')(audio_c2_s)

    audio_c2 = Multiply()([audio_c2_l, audio_c2_s])  # (N, 15, 32, 32)
    audio_p2 = audio_pooling_layer(audio_c2)
    return audio_p2


def audio_cnn_layer3(audio_p2):
    audio_c3_l = layers.Conv2D(32, (3, 5), padding='same')(audio_p2)  # (N, 15, 16, 32)
    audio_c3_l = layers.BatchNormalization()(audio_c3_l)
    audio_c3_l = Activation('linear')(audio_c3_l)

    audio_c3_s = layers.Conv2D(32, (3, 5), padding='same')(audio_p2)  # (N, 15, 16, 32)
    audio_c3_s = layers.BatchNormalization()(audio_c3_s)
    audio_c3_s = Activation('sigmoid')(audio_c3_s)

    audio_c3 = Multiply()([audio_c3_l, audio_c3_s])  # (N, 15, 16, 32)
    audio_p3 = audio_pooling_layer(audio_c3)
    return audio_p3


def audio_cnn_layer4(audio_p3):
    audio_c4_l = layers.Conv2D(32, (3, 5), padding='same')(audio_p3)  # (N, 15, 8, 32)
    audio_c4_l = layers.BatchNormalization()(audio_c4_l)
    audio_c4_l = Activation('linear')(audio_c4_l)

    audio_c4_s = layers.Conv2D(32, (3, 5), padding='same')(audio_p3)  # (N, 15, 8, 32)
    audio_c4_s = layers.BatchNormalization()(audio_c4_s)
    audio_c4_s = Activation('sigmoid')(audio_c4_s)

    audio_c4 = Multiply()([audio_c4_l, audio_c4_s])  # (N, 15, 16, 32)
    audio_p4 = audio_pooling_layer(audio_c4)
    return audio_p4


def audio_middle_part(audio_part_in):
    audio_part_in = layers.BatchNormalization()(audio_part_in)
    audio_part_in = layers.ReLU()(audio_part_in)  # (None, 15, 4, 16)
    audio_part_in = layers.Conv2D(16, (5, 3), strides=(3, 1), padding='same')(audio_part_in)  # (None, 15, 4, 16)
    audio_part_in = layers.BatchNormalization()(audio_part_in)
    audio_part_in = layers.ReLU()(audio_part_in)  # (None, 5, 4, 16)
    audio_part_in = Flatten()(audio_part_in)  # 320
    return audio_part_in


def video_cnn_layer1(video_input):
    video_c1_l = layers.Conv2D(32, (5, 13), padding='same')(video_input)  # (N, 450, 300, 32)
    video_c1_l = layers.BatchNormalization()(video_c1_l)
    video_c1_l = Activation('linear')(video_c1_l)

    video_c1_s = layers.Conv2D(32, (5, 13), padding='same')(video_input)  # (N, 450, 300, 32)
    video_c1_s = layers.BatchNormalization()(video_c1_s)
    video_c1_s = Activation('sigmoid')(video_c1_s)

    video_c1 = Multiply()([video_c1_l, video_c1_s])

    video_p1 = layers.Conv2D(32, (5, 5), strides=(3, 3), padding='same')(video_c1)  # (None, 150, 100, 32)
    video_p1 = layers.BatchNormalization()(video_p1)
    video_p1 = layers.ReLU()(video_p1)
    return video_p1


def video_cnn_layer2(video_p1):
    video_c2_l = layers.Conv2D(32, (5, 13), padding='same')(video_p1)
    video_c2_l = layers.BatchNormalization()(video_c2_l)
    video_c2_l = Activation('linear')(video_c2_l)

    video_c2_s = layers.Conv2D(32, (5, 13), padding='same')(video_p1)
    video_c2_s = layers.BatchNormalization()(video_c2_s)
    video_c2_s = Activation('sigmoid')(video_c2_s)

    video_c2 = Multiply()([video_c2_l, video_c2_s])  # (N, 15, 32, 32)

    video_p2 = layers.Conv2D(32, (5, 5), strides=(3, 3), padding='same')(video_c2)  # (None, 75, 50, 32)
    video_p2 = layers.BatchNormalization()(video_p2)
    video_p2 = layers.ReLU()(video_p2)
    return video_p2


def video_cnn_layer3(video_p2):
    video_c3_l = layers.Conv2D(32, (3, 11), padding='same')(video_p2)  # (None, 75, 50, 32)
    video_c3_l = layers.BatchNormalization()(video_c3_l)
    video_c3_l = Activation('linear')(video_c3_l)

    video_c3_s = layers.Conv2D(32, (3, 11), padding='same')(video_p2)  # (None, 75, 50, 32)
    video_c3_s = layers.BatchNormalization()(video_c3_s)
    video_c3_s = Activation('sigmoid')(video_c3_s)

    video_c3 = Multiply()([video_c3_l, video_c3_s])  # (N, 15, 16, 32)

    video_p3 = layers.Conv2D(32, (3, 3), strides=(2, 2), padding='same')(video_c3)  # (None, 25, 17, 32)
    # 75+5-1 = 69/2 = 35
    video_p3 = layers.BatchNormalization()(video_p3)
    video_p3 = layers.ReLU()(video_p3)
    return video_p3


def video_cnn_layer4(video_p3):
    video_c4_l = layers.Conv2D(32, (3, 7), padding='same')(video_p3)  # (None, 35, 29, 32)
    video_c4_l = layers.BatchNormalization()(video_c4_l)
    video_c4_l = Activation('linear')(video_c4_l)

    video_c4_s = layers.Conv2D(32, (3, 7), padding='same')(video_p3)  # (None, 35, 29, 32)
    video_c4_s = layers.BatchNormalization()(video_c4_s)
    video_c4_s = Activation('sigmoid')(video_c4_s)

    video_c4 = Multiply()([video_c4_l, video_c4_s])  # (N, 15, 16, 32)

    video_p4 = layers.Conv2D(32, (3, 3), strides=(2, 2), padding='same')(video_c4)  # (None, 13, 9, 32)
    video_p4 = layers.BatchNormalization()(video_p4)
    video_p4 = layers.ReLU()(video_p4)
    return video_p4


def video_middle_part(video_part_in):
    video_part_in = layers.BatchNormalization()(video_part_in)
    video_part_in = layers.ReLU()(video_part_in)  # (None, 15, 4, 16)
    video_part_in = layers.Conv2D(16, (3, 3), strides=(2, 1), padding='same')(video_part_in)  # (None, 15, 4, 16)
    video_part_in = layers.BatchNormalization()(video_part_in)
    video_part_in = layers.ReLU()(video_part_in)  # (None, 5, 4, 16)
    video_part_in = Flatten()(video_part_in)  # 320
    return video_part_in


def average_pooling(video_close_glu, name):
    video_close_GAP_input = Reshape(target_shape=(int(video_close_glu.get_shape()[1]), 1))(video_close_glu)
    video_close_output = GlobalAveragePooling1D(name=name)(video_close_GAP_input)
    return video_close_output


def model_rulenet(audio, image, audio_time, audio_freq, video_height, video_width, video_input_frames_num, bi_gru=False):

    if audio:
        audio_input = Input(shape=(audio_time, audio_freq, 1), name='audio_in')  # (N, 15, 64)

        audio_p1 = audio_cnn_layer1(audio_input)
        audio_p2 = audio_cnn_layer2(audio_p1)
        audio_p3 = audio_cnn_layer3(audio_p2)
        audio_p4 = audio_cnn_layer4(audio_p3)

        # audio singing
        audio_singing = layers.Conv2D(16, (1, 1), strides=(1, 1), padding='same', name='audio_singing')(audio_p4)
        audio_singing = layers.BatchNormalization()(audio_singing)
        audio_singing = layers.ReLU()(audio_singing)  # (None, 15, 4, 16)

        rnn_size = 64
        conv_shape = audio_singing.get_shape()
        a1 = Reshape(target_shape=(int(conv_shape[1]), int(conv_shape[2] * conv_shape[3])))(audio_singing)
        if bi_gru:
            a1_rnn = Bidirectional(GRU(rnn_size, name='audio_singing_rnn'))(a1)  # (None, 128)
        else:
            a1_rnn = GRU(rnn_size, name='audio_singing_rnn')(a1)
        a1_rnnout = layers.BatchNormalization()(a1_rnn)
        a1_rnnout = Activation('linear', name='audio_singing_rnn_linear')(a1_rnnout)
        if bi_gru:
            a1_rnnout_gate = Bidirectional(GRU(rnn_size, name='audio_singing_rnn_gate'))(a1)  # (None, 128)
        else:
            a1_rnnout_gate = GRU(rnn_size, name='audio_singing_rnn_gate')(a1)
        a1_rnnout_gate_out = layers.BatchNormalization()(a1_rnnout_gate)
        a1_rnnout_gate_out = Activation('sigmoid', name='audio_singing_rnn_sigmoid')(a1_rnnout_gate_out)
        audio_singing = Multiply()([a1_rnnout, a1_rnnout_gate_out])

        audio_singing_s = Dense(128, name='audio_singing32_s')(audio_singing)  # (N, 128)
        audio_singing_s = layers.BatchNormalization()(audio_singing_s)
        audio_singing_s = Activation('sigmoid')(audio_singing_s)
        audio_singing_glu = audio_singing_s  # (N, 128) # 这个128维的glu是要用到后面共享融合的

        audio_singing_output = Dense(1, activation='sigmoid', name='audio_singing_output')(audio_singing_glu)

        # audio speech
        audio_speech = layers.Conv2D(16, (1, 1), strides=(1, 1), padding='same', name='audio_speech')(audio_p4)
        audio_speech = layers.BatchNormalization()(audio_speech)
        audio_speech = layers.ReLU()(audio_speech)  # (None, 15, 4, 16)

        rnn_size = 64
        conv_shape = audio_speech.get_shape()
        a1 = Reshape(target_shape=(int(conv_shape[1]), int(conv_shape[2] * conv_shape[3])))(audio_speech)
        if bi_gru:
            a1_rnn = Bidirectional(GRU(rnn_size, name='audio_speech_rnn'))(a1)  # (None, 128)
        else:
            a1_rnn = GRU(rnn_size, name='audio_speech_rnn')(a1)
        a1_rnnout = layers.BatchNormalization()(a1_rnn)
        a1_rnnout = Activation('linear', name='audio_speech_rnn_linear')(a1_rnnout)
        if bi_gru:
            a1_rnnout_gate = Bidirectional(GRU(rnn_size, name='audio_speech_rnn_gate'))(a1)  # (None, 128)
        else:
            a1_rnnout_gate = GRU(rnn_size, name='audio_speech_rnn_gate')(a1)
        a1_rnnout_gate_out = layers.BatchNormalization()(a1_rnnout_gate)
        a1_rnnout_gate_out = Activation('sigmoid', name='audio_speech_rnn_sigmoid')(a1_rnnout_gate_out)
        audio_speech = Multiply()([a1_rnnout, a1_rnnout_gate_out])

        audio_speech_s = Dense(128, name='audio_speech32_s')(audio_speech)  # (N, 128)
        audio_speech_s = layers.BatchNormalization()(audio_speech_s)
        audio_speech_s = Activation('sigmoid')(audio_speech_s)
        audio_speech_glu = audio_speech_s  # (N, 128) # 这个128维的glu是要用到后面共享融合的

        audio_speech_output = Dense(1, activation='sigmoid', name='audio_speech_output')(audio_speech_glu)

        # audio others
        audio_others = layers.Conv2D(16, (1, 1), strides=(1, 1), padding='same', name='audio_others')(audio_p4)
        audio_others = layers.BatchNormalization()(audio_others)
        audio_others = layers.ReLU()(audio_others)  # (None, 15, 4, 16)

        rnn_size = 64
        conv_shape = audio_others.get_shape()
        a1 = Reshape(target_shape=(int(conv_shape[1]), int(conv_shape[2] * conv_shape[3])))(audio_others)
        if bi_gru:
            a1_rnn = Bidirectional(GRU(rnn_size, name='audio_others_rnn'))(a1)  # (None, 128)
        else:
            a1_rnn = GRU(rnn_size, name='audio_others_rnn')(a1)
        a1_rnnout = layers.BatchNormalization()(a1_rnn)
        a1_rnnout = Activation('linear', name='audio_others_rnn_linear')(a1_rnnout)
        if bi_gru:
            a1_rnnout_gate = Bidirectional(GRU(rnn_size, name='audio_others_rnn_gate'))(a1)  # (None, 128)
        else:
            a1_rnnout_gate = GRU(rnn_size, name='audio_others_rnn_gate')(a1)
        a1_rnnout_gate_out = layers.BatchNormalization()(a1_rnnout_gate)
        a1_rnnout_gate_out = Activation('sigmoid', name='audio_others_rnn_sigmoid')(a1_rnnout_gate_out)
        audio_others = Multiply()([a1_rnnout, a1_rnnout_gate_out])

        audio_others_s = Dense(128, name='audio_others32_s')(audio_others)  # (N, 128)
        audio_others_s = layers.BatchNormalization()(audio_others_s)
        audio_others_s = Activation('sigmoid')(audio_others_s)
        audio_others_glu = audio_others_s  # (N, 128) # 这个128维的glu是要用到后面共享融合的

        audio_others_output = Dense(1, activation='sigmoid', name='audio_others_output')(audio_others_glu)

        # audio silence
        audio_silence = layers.Conv2D(16, (1, 1), strides=(1, 1), padding='same', name='audio_silence')(audio_p4)
        audio_silence = layers.BatchNormalization()(audio_silence)
        audio_silence = layers.ReLU()(audio_silence)  # (None, 15, 4, 16)

        rnn_size = 64
        conv_shape = audio_silence.get_shape()
        a1 = Reshape(target_shape=(int(conv_shape[1]), int(conv_shape[2] * conv_shape[3])))(audio_silence)
        if bi_gru:
            a1_rnn = Bidirectional(GRU(rnn_size, name='audio_silence_rnn'))(a1)  # (None, 128)
        else:
            a1_rnn = GRU(rnn_size, name='audio_silence_rnn')(a1)
        a1_rnnout = layers.BatchNormalization()(a1_rnn)
        a1_rnnout = Activation('linear', name='audio_silence_rnn_linear')(a1_rnnout)
        if bi_gru:
            a1_rnnout_gate = Bidirectional(GRU(rnn_size, name='audio_silence_rnn_gate'))(a1)  # (None, 128)
        else:
            a1_rnnout_gate = GRU(rnn_size, name='audio_silence_rnn_gate')(a1)
        a1_rnnout_gate_out = layers.BatchNormalization()(a1_rnnout_gate)
        a1_rnnout_gate_out = Activation('sigmoid', name='audio_silence_rnn_sigmoid')(a1_rnnout_gate_out)
        audio_silence = Multiply()([a1_rnnout, a1_rnnout_gate_out])

        audio_silence_s = Dense(128, name='audio_silence32_s')(audio_silence)  # (N, 128)
        audio_silence_s = layers.BatchNormalization()(audio_silence_s)
        audio_silence_s = Activation('sigmoid')(audio_silence_s)
        audio_silence_glu = audio_silence_s  # (N, 128) # 这个128维的glu是要用到后面共享融合的

        audio_silence_output = Dense(1, activation='sigmoid', name='audio_silence_output')(audio_silence_glu)


    if image:
        video_input = Input(shape=(video_height, video_width, video_input_frames_num), name='video_in')

        video_p1 = video_cnn_layer1(video_input)
        video_p2 = video_cnn_layer2(video_p1)
        video_p3 = video_cnn_layer3(video_p2)
        video_p4 = video_cnn_layer4(video_p3)

        # video open
        video_open = layers.Conv2D(16, (1, 1), strides=(1, 1), padding='same', name='video_open')(video_p4)
        video_open = layers.BatchNormalization()(video_open)
        video_open = layers.ReLU()(video_open)  # (None, 15, 4, 16)
        video_open = layers.Conv2D(16, (3, 3), strides=(2, 1), padding='same')(video_open)  # (None, 15, 4, 16)
        video_open = layers.BatchNormalization()(video_open)
        video_open = layers.ReLU()(video_open)  # (None, 5, 4, 16)
        video_open = Flatten()(video_open)  # 320

        video_open_s = Dense(128, name='video_open32_s')(video_open)  # (N, 128)
        video_open_s = layers.BatchNormalization()(video_open_s)
        video_open_s = Activation('sigmoid')(video_open_s)
        video_open_glu = video_open_s  # (N, 128) # 这个128维的glu是要用到后面共享融合的

        video_open_output = Dense(1, activation='sigmoid', name='video_open_output')(video_open_glu)

        # video close
        video_close = layers.Conv2D(16, (1, 1), strides=(1, 1), padding='same', name='video_close')(video_p4)
        video_close = layers.BatchNormalization()(video_close)
        video_close = layers.ReLU()(video_close)  # (None, 15, 4, 16)
        video_close = layers.Conv2D(16, (3, 3), strides=(2, 1), padding='same')(video_close)  # (None, 15, 4, 16)
        video_close = layers.BatchNormalization()(video_close)
        video_close = layers.ReLU()(video_close)  # (None, 5, 4, 16)
        video_close = Flatten()(video_close)  # 320

        video_close_s = Dense(128, name='video_close32_s')(video_close)  # (N, 128)
        video_close_s = layers.BatchNormalization()(video_close_s)
        video_close_s = Activation('sigmoid')(video_close_s)
        video_close_glu = video_close_s  # (N, 128) # 这个128维的glu是要用到后面共享融合的

        video_close_output = Dense(1, activation='sigmoid', name='video_close_output')(video_close_glu)

    if audio and image:
        multiply_singing = Multiply()([audio_singing_glu, video_open_glu])
        vector_singing = Dense(256, name='vector_singing')(multiply_singing)
        vector_singing = layers.BatchNormalization()(vector_singing)
        vector_singing = layers.ReLU()(vector_singing)
        joint_singing_out = Dense(1, activation='sigmoid', name='joint_singing_out')(vector_singing)

        multiply_speech = Multiply()([audio_speech_glu, video_open_glu])
        vector_speech = Dense(256, name='vector_speech')(multiply_speech)
        vector_speech = layers.BatchNormalization()(vector_speech)
        vector_speech = layers.ReLU()(vector_speech)
        joint_speech_out = Dense(1, activation='sigmoid', name='joint_speech_out')(vector_speech)

        multiply_others_1 = Multiply()([audio_others_glu, video_open_glu])
        multiply_others_2 = Multiply()([audio_others_glu, video_close_glu])
        multiply_others_3 = Multiply()([audio_singing_glu, video_close_glu])
        multiply_others_4 = Multiply()([audio_speech_glu, video_close_glu])
        concat_others = Concatenate()([multiply_others_1, multiply_others_2, multiply_others_3, multiply_others_4])
        vector_others = Dense(256, name='vector_others')(concat_others)
        vector_others = layers.BatchNormalization()(vector_others)
        vector_others = layers.ReLU()(vector_others)
        joint_others_out = Dense(1, activation='sigmoid', name='joint_others_out')(vector_others)

        multiply_silence_1 = Multiply()([audio_silence_glu, video_open_glu])
        multiply_silence_2 = Multiply()([audio_silence_glu, video_close_glu])
        concat_silence = Concatenate()([multiply_silence_1, multiply_silence_2])
        vector_silence = Dense(256, name='vector_silence')(concat_silence)
        vector_silence = layers.BatchNormalization()(vector_silence)
        vector_silence = layers.ReLU()(vector_silence)
        joint_silence_out = Dense(1, activation='sigmoid', name='joint_silence_out')(vector_silence)

    if audio and image:
        model = Model(inputs=[audio_input, video_input],
                      outputs=[audio_singing_output, audio_speech_output, audio_silence_output, audio_others_output,
                               video_open_output, video_close_output,
                               joint_singing_out, joint_speech_out, joint_silence_out, joint_others_out])
    elif audio:
        model = Model(inputs=[audio_input],
                      outputs=[audio_singing_output, audio_speech_output, audio_silence_output, audio_others_output, ])

    model.summary()

    if audio and image:
        joint_independent_compile(model)
    elif audio:
        audio_independent_compile(model)
    return model
