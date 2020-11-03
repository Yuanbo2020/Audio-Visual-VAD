#coding= utf-8
import h5py, os
import framework.config as config
from framework.keras_data_generator import DataGenerator_2D_inputframes
from framework.model_architecture import *

def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)


from keras.callbacks import *
class hyb_ModelCheckpoint(Callback):
    def __init__(self, filepath,
                 monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1, start_epoch=None):
        super(hyb_ModelCheckpoint, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0
        self.start_epoch = start_epoch

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure') or 'binary_accuracy' in self.monitor:
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        if self.start_epoch:
            epoch += self.start_epoch
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            # print(logs)
            # {'val_loss': 0.2926377449068865, 'val_binary_accuracy': 0.9926498422455728,
            # 'loss': 0.45031250206433826, 'binary_accuracy': 0.9397529899686968}
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            if self.start_epoch:
                print('\ncontinue training current epoch:', epoch)
                print('\nsave path:', filepath)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch + 1, self.monitor, self.best,
                                     current, filepath), flush=True)
                        self.best = current
                        if self.save_weights_only:
                            self.model.save_weights(filepath, overwrite=True)
                        else:
                            self.model.save(filepath, overwrite=True)
                    else:
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s did not improve' %
                                  (epoch + 1, self.monitor), flush=True)
            else:
                if self.verbose > 0:
                    print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath), flush=True)
                if self.save_weights_only:
                    self.model.save_weights(filepath, overwrite=True)
                else:
                    self.model.save(filepath, overwrite=True)


def load_data_by_each_frame(aggregator_path, audio, image, sampling_suffix, regularize_frame_num):
    print('aggregator_path:', aggregator_path)
    filenames = [file for file in os.listdir(aggregator_path) if 'audio_x_normal_3fp' in file]

    training_filenames = []
    validation_filenames = []

    for file in filenames:
        part = file.split('audio_x_normal' + sampling_suffix)[0]
        if 'training' in file and part not in training_filenames:
            training_filenames.append(part)
        elif 'validation' in file and part not in validation_filenames:
            validation_filenames.append(part)
    print('training files: %d , validation files: %d '%(len(training_filenames), len(validation_filenames)))


    train_x_audio, train_y_audio, train_y, training_images_path_list = load_files(training_filenames, audio, image,
                                                                   sampling_suffix, aggregator_path,
                                                                   regularize_frame_num)
    val_x_audio, val_y_audio, val_y, val_images_path_list = load_files(validation_filenames, audio, image,
                                                                   sampling_suffix, aggregator_path,
                                                                   regularize_frame_num)

    if audio and image:
        print('training:', len(training_filenames), train_x_audio.shape, len(training_images_path_list), train_y.shape, flush=True)
        print('validation:', len(validation_filenames), val_x_audio.shape, len(val_images_path_list), val_y.shape, flush=True)
        return train_x_audio, training_images_path_list, train_y_audio, train_y, \
               val_x_audio, val_images_path_list, val_y_audio, val_y
    elif audio:
        print('training:', len(training_filenames), train_x_audio.shape, train_y.shape, flush=True)
        print('validation:', len(validation_filenames), val_x_audio.shape, val_y.shape, flush=True)
        return train_x_audio, None, train_y, val_x_audio, None, val_y


def train_keras_generator_read_by_each_picture(model_num,
                                               model_dir,
                                               imageflag,
                                               audioflag,
                                               aggregator_path,
                                               label_dependent,
                                               regularize_frame_num,
                                               continue_training=config.continue_training,
                                               sampling_suffix=config.sampling_suffix,
                                               image_normalization=config.image_normalization):
    np.random.seed(config.random_seed)

    train_x_audio, training_images_path_list, train_y_audio, train_y, \
    val_x_audio, validation_images_path_list, val_y_audio, val_y = load_data_by_each_frame(aggregator_path,
                                                                                audioflag,
                                                                                imageflag,
                                                                                sampling_suffix,
                                                                                regularize_frame_num)

    audio_time, audio_freq, video_height, video_width = 0, 0, 0, 0
    if audioflag:
        (_, audio_time, audio_freq) = train_x_audio.shape
    if imageflag:
        video_height, video_width = config.image_box_height960_width544[0], config.image_box_height960_width544[1]
    (_, num_classes) = train_y.shape

    # Generators
    training_generator = DataGenerator_2D_inputframes(train_x_audio, training_images_path_list, train_y_audio, train_y,
                                             config.batch, audioflag, imageflag, image_normalization, label_dependent,
                                                      regularize_frame_num)
    validation_generator = DataGenerator_2D_inputframes(val_x_audio, validation_images_path_list, val_y_auido, val_y,
                                               config.batch, audioflag, imageflag, image_normalization, label_dependent,
                                                        regularize_frame_num)

    print('model_dir', model_dir)
    if not continue_training:
        model = model_rulenet(audioflag, imageflag, audio_time, audio_freq, video_height, video_width, video_input_frames_num)
    else:
        model_list = [file for file in os.listdir(model_dir) if file.endswith('.hdf5')]
        max_epoch_num = 0
        for epoch_model in model_list:
            epoch_num = int(epoch_model.split('-')[0].split('_')[-1])
            if epoch_num > max_epoch_num:
                max_epoch_num = epoch_num
                max_epoch_model = epoch_model
        previous_model_path = os.path.join(model_dir, max_epoch_model)
        print('previous_model_path:', previous_model_path)
        model = load_model(previous_model_path)
        start_epoch = max_epoch_num

    create_folder(model_dir)

    last_layer_name = 'joint_' + str(model_num)

    if audioflag and imageflag:
        if label_dependent:
            filepath = os.path.join(model_dir,
                                    'model_{epoch:02d}-{loss:.4f}-{' + last_layer_name + '_loss:.4f}'\
                                    '-{audio_final_out_loss:.4f}-{video_final_out_loss:.4f}'\
                                    '-{val_loss:.4f}-{val_' + last_layer_name + '_loss:.4f}'\
                                    '-{val_audio_final_out_loss:.4f}-{val_video_final_out_loss:.4f}.hdf5')
        else:
            filepath = os.path.join(model_dir, 'model_{epoch:02d}-{loss:.4f}-'
                                               '{audio_singing_output_loss:.4f}-'
                                               '{audio_speech_output_loss:.4f}-'
                                               '{audio_silence_output_loss:.4f}-'
                                               '{audio_others_output_loss:.4f}-'
                                               '{video_open_output_loss:.4f}-'
                                               '{video_close_output_loss:.4f}-'
                                               '{joint_singing_out_loss:.4f}-'
                                               '{joint_speech_out_loss:.4f}-'
                                               '{joint_silence_out_loss:.4f}-'
                                               '{joint_others_out_loss:.4f}-'
                                               '{val_loss:.4f}-'
                                               '{val_audio_singing_output_loss:.4f}-'
                                               '{val_audio_speech_output_loss:.4f}-'
                                               '{val_audio_silence_output_loss:.4f}-'
                                               '{val_audio_others_output_loss:.4f}-'
                                               '{val_video_open_output_loss:.4f}-'
                                               '{val_video_close_output_loss:.4f}-'
                                               '{val_joint_singing_out_loss:.4f}-'
                                               '{val_joint_speech_out_loss:.4f}-'
                                               '{val_joint_silence_out_loss:.4f}-'
                                               '{val_joint_others_out_loss:.4f}.hdf5')

    elif audioflag:
        if label_dependent:
            filepath = os.path.join(model_dir, 'model_{epoch:02d}-{loss:.4f}-{val_loss:.4f}.hdf5')
        else:
            filepath = os.path.join(model_dir, 'model_{epoch:02d}-{loss:.4f}-'
                                               '{audio_singing_output_loss:.4f}-'
                                               '{audio_speech_output_loss:.4f}-'
                                               '{audio_silence_output_loss:.4f}-'
                                               '{audio_others_output_loss:.4f}-'
                                               '{val_loss:.4f}-'
                                               '{val_audio_singing_output_loss:.4f}-'
                                               '{val_audio_speech_output_loss:.4f}-'
                                               '{val_audio_silence_output_loss:.4f}-'
                                               '{val_audio_others_output_loss:.4f}.hdf5')

    if continue_training:
        print('filepath:', filepath)
        save_model = hyb_ModelCheckpoint(filepath=filepath,  monitor='loss', verbose=0,
                                         save_best_only=False, save_weights_only=False, mode='min', period=1,
                                         start_epoch=start_epoch)
    else:
        save_model = hyb_ModelCheckpoint(filepath=filepath, monitor='loss', verbose=0,
                                         save_best_only=False, save_weights_only=False, mode='min', period=1)

    print('batch:', config.batch, flush=True)

    if continue_training:
        epoch = config.epoch - start_epoch
    else:
        epoch = config.epoch

    hist = model.fit_generator(training_generator,
                               steps_per_epoch=int(np.ceil(len(train_y)/config.batch)),
                               epochs=epoch,
                               verbose=1,
                               callbacks=[save_model],
                               validation_data=validation_generator,
                               validation_steps=int(np.ceil(len(val_y)/config.batch)),
                               class_weight=None,
                               max_queue_size=10,
                               workers=1,
                               initial_epoch=0)

    print('hist.history:', hist.history, flush=True)

    log_file = os.path.join(model_dir, 'log.txt')
    with open(log_file, 'w') as f:
        f.write(str(hist.history))









