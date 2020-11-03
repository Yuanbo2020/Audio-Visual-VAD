import numpy as np
import cv2
import framework.config as config

np.random.seed(config.random_seed)  # for reproducibility


def convert_image_to_inputframes(train_x_video, video_height, video_width, regularize_frame_num):
    # print('input shape:', train_x_video.shape)
    train_x_video = train_x_video.reshape((-1, regularize_frame_num, video_height, video_width))
    # （高度，宽度，深度）
    train_x_video = train_x_video.transpose(0, 2, 3, 1)
    # print('3D shape:', train_x_video.shape)
    return train_x_video

def make_video_y(train_y, self_attention=None):
    # label_list = ['singing', 'speech', 'silence', 'others']
    if self_attention:
        video_y = np.zeros((train_y.shape[0], 1))
    else:
        video_y = np.zeros((train_y.shape[0], 2))

    for i in range(train_y.shape[0]):
        one_position = list(train_y[i]).index(1)  # 0,1, 2,3
        if self_attention:
            if one_position < 2:  # 0, 1 说明开口
                video_y[i, 0] = 1
            else:
                video_y[i, 0] = 0
        else:
            if one_position < 2:  # 0, 1 说明开口
                video_y[i, 0] = 1  # 开、闭
            else:
                video_y[i, 1] = 1
    return video_y


def get_array(array):
    return array.reshape(len(array), 1)


def DataGenerator_2D_inputframes(train_x_audio, training_images_list, train_y_audio, train_y, batch_size, audio, image,
                                 image_normalization, label_dependent, regularize_frame_num):

    indices = np.arange(len(train_y))
    while True:
        if not train_y is None:
            np.random.shuffle(indices)

        for i in range(int(np.ceil(len(train_y)/batch_size))):
            if i * batch_size + batch_size > len(train_y) - 1:
                excerpt = indices[i * batch_size:]
            else:
                excerpt = indices[i * batch_size: i * batch_size + batch_size]

            if image:
                images_list = []
                for num, excerpt_id in enumerate(excerpt):
                    images_list.extend(training_images_list[excerpt_id])

                train_x_video = []
                for image_path in images_list:
                    img = cv2.imread(image_path, 0)

                    # 有归一化相比于没有，整体有1%左右的提升
                    if image_normalization:
                        img = (img - np.min(img)) / (np.max(img) - np.min(img))

                    train_x_video.append(img)
                train_x_video = np.array(train_x_video)
                video_height, video_width = config.image_box_height960_width544[0], \
                                            config.image_box_height960_width544[1]
                train_x_video = convert_image_to_inputframes(train_x_video, video_height, video_width, regularize_frame_num)

                video_y = make_video_y(train_y[excerpt], self_attention)

            if audio and image:
                if label_dependent:
                    yield [train_x_audio[excerpt], train_x_video], \
                          [train_y_audio[excerpt], train_y[excerpt], video_y, train_y[excerpt]]
                else:
                    yield [train_x_audio[excerpt], train_x_video], \
                          [get_array(train_y_audio[excerpt][:, 0]),
                           get_array(train_y_audio[excerpt][:, 1]),
                           get_array(train_y_audio[excerpt][:, 2]),
                           get_array(train_y_audio[excerpt][:, 3]),

                           get_array(train_y[excerpt][:, 0]),
                           get_array(train_y[excerpt][:, 1]),
                           get_array(train_y[excerpt][:, 2]),
                           get_array(train_y[excerpt][:, 3]),

                           get_array(video_y[:, 0]),
                           get_array(video_y[:, 1]),

                           get_array(train_y[excerpt][:, 0]),
                           get_array(train_y[excerpt][:, 1]),
                           get_array(train_y[excerpt][:, 2]),
                           get_array(train_y[excerpt][:, 3])]
            elif audio:
                if label_dependent:  # 如果标签之间是相互依赖的整体
                    yield train_x_audio[excerpt], train_y_audio[excerpt]
                else:
                    yield [train_x_audio[excerpt]], \
                          [get_array(train_y_audio[excerpt][:, 0]),
                           get_array(train_y_audio[excerpt][:, 1]),
                           get_array(train_y_audio[excerpt][:, 2]),
                           get_array(train_y_audio[excerpt][:, 3])]


