# -*- coding:utf-8 -*-
from __future__ import absolute_import, print_function

import warnings
from keras.utils import plot_model
import numpy as np
from keras import backend as K
from keras import layers
from keras.applications.imagenet_utils import decode_predictions
from keras.engine.topology import get_source_inputs
from keras.layers import (Activation, AveragePooling2D, BatchNormalization,
                          Conv2D, Dense, GlobalAveragePooling2D, Flatten,
                          GlobalMaxPooling2D, Input, MaxPooling2D, ZeroPadding2D)
from keras.models import Model
from keras.preprocessing import image
from keras.utils.data_utils import get_file
from keras.utils.layer_utils import convert_all_kernels_in_model
import matplotlib.pyplot as plt

WEIGHTS_PATH = ('https://github.com/fchollet/deep-learning-models/'
                'releases/download/v0.2/'
                'resnet50_weights_tf_dim_ordering_tf_kernels.h5')

def identify_block(input_tensor, kernel_size, filters, stage, block):
    """The identity block is the block that has no conv layer at shortcut."""
    filters1, filters2, filters3 = filters

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # 减少通道数
    x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)
    # 3x3卷积
    x = Conv2D(filters2, kernel_size, padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)
    # 上升通道数
    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(name=bn_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    """A block that has a conv layer at shortcut.
        Note that from stage 3,
        the first conv layer at main path is with strides=(2, 2)
        And the shortcut should have strides=(2, 2) as well.
    """
    filters1, filters2, filters3 = filters

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # 减少通道数
    x = Conv2D(filters1, (1, 1), strides=strides, name=conv_name_base + '_2a')(input_tensor)
    x = BatchNormalization(name=bn_name_base + '_2a')(x)
    x = Activation('relu')(x)
    # 3x3卷积
    x = Conv2D(filters2, kernel_size, padding='same', name=conv_name_base + '_2b')(x)
    x = BatchNormalization(name=bn_name_base + '_2b')(x)
    x = Activation('relu')(x)
    # 上升通道数
    x = Conv2D(filters3, (1, 1), name=conv_name_base + '_2c')(x)
    x = BatchNormalization(name=bn_name_base + '_2c')(x)
    # 残差边
    shortcut = Conv2D(filters3, (1, 1), strides=strides, name=conv_name_base + '_sc')(input_tensor)
    shortcut = BatchNormalization(name=bn_name_base + '_sc')(shortcut)

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x


def resnet50(input_shape=[224, 224, 3], classes=1000):
    img_input = Input(shape=input_shape)
    # (None, 224, 224, 3) -> (None, 230, 230, 3)
    x = ZeroPadding2D((3, 3), name='conv1_pad')(img_input)  # 表示上下各填充3行0

    # (None, 230, 230, 3) -> (None, 112, 112, 64)
    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(x)
    x = BatchNormalization(name='bn_conv1')(x)
    x = Activation('relu')(x)

    x = ZeroPadding2D((1, 1), name='pool1_pad')(x)  # 表示上下各填充1行0
    # (None, 112, 112, 64) -> (None, 56, 56, 64)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    # 56,56,64 -> 56,56,256
    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identify_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identify_block(x, 3, [64, 64, 256], stage=2, block='c')

    # (None, 56, 56, 256) -> (None, 28, 28, 512)
    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identify_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identify_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identify_block(x, 3, [128, 128, 512], stage=3, block='d')

    # (None, 28, 28, 512) -> (None, 14, 14, 1024)
    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identify_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identify_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identify_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identify_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identify_block(x, 3, [256, 256, 1024], stage=4, block='f')

    # (None, 14, 14, 1024) ->  (None, 7, 7, 2048)
    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identify_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identify_block(x, 3, [512, 512, 2048], stage=5, block='c')

    # 1, 1, 2048
    x = AveragePooling2D((7, 7), name='avg_pool')(x)
    # 2048
    x = Flatten()(x)

    x = Dense(classes, activation='softmax', name='fc1000')(x)

    model = Model(img_input, x, name='resnet50')
    return model


if __name__ == "__main__":
    model = resnet50()
    plot_model(model, "resnet50.svg", show_shapes=True)
    # model.summary()
    weights_path = get_file('resnet50_weights_tf_dim_ordering_tf_kernels.h5', WEIGHTS_PATH)
    model.load_weights(weights_path)

    img_path = "elephant.jpg"
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    print('Input image shape:', x.shape)

    preds = model.predict(x)
    pred_label = decode_predictions(preds)[0][0][1]
    # print('Predicted:', decode_predictions(preds))

    plt.imshow(img)
    plt.title('Predicted: ' + str(pred_label), loc='center')
    plt.savefig('predicted.png')
    plt.show()