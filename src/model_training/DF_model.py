# The model is the DF model by Sirinam et al

from keras.models import Model
from keras.layers import Dense
from keras.layers import Input

from keras.layers import Activation
from keras.layers import ELU
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.layers import Dropout
from keras.layers.core import Flatten


def DF(input_shape=None, emb_size=None):
    # -----------------Entry flow -----------------
    input_data = Input(shape=input_shape)

    filter_num = ['None', 32, 64, 128, 256]
    kernel_size = ['None', 8, 8, 8, 8]
    conv_stride_size = ['None', 1, 1, 1, 1]
    pool_stride_size = ['None', 4, 4, 4, 4]
    pool_size = ['None', 8, 8, 8, 8]

    model = Conv1D(filters=filter_num[1], kernel_size=kernel_size[1],
                   strides=conv_stride_size[1], padding='same', name='block1_conv1')(input_data)
    model = ELU(alpha=1.0, name='block1_adv_act1')(model)
    model = Conv1D(filters=filter_num[1], kernel_size=kernel_size[1],
                   strides=conv_stride_size[1], padding='same', name='block1_conv2')(model)
    model = ELU(alpha=1.0, name='block1_adv_act2')(model)
    model = MaxPooling1D(pool_size=pool_size[1], strides=pool_stride_size[1],
                         padding='same', name='block1_pool')(model)
    model = Dropout(0.1, name='block1_dropout')(model)

    model = Conv1D(filters=filter_num[2], kernel_size=kernel_size[2],
                   strides=conv_stride_size[2], padding='same', name='block2_conv1')(model)
    model = Activation('relu', name='block2_act1')(model)
    model = Conv1D(filters=filter_num[2], kernel_size=kernel_size[2],
                   strides=conv_stride_size[2], padding='same', name='block2_conv2')(model)
    model = Activation('relu', name='block2_act2')(model)
    model = MaxPooling1D(pool_size=pool_size[2], strides=pool_stride_size[3],
                         padding='same', name='block2_pool')(model)
    model = Dropout(0.1, name='block2_dropout')(model)

    model = Conv1D(filters=filter_num[3], kernel_size=kernel_size[3],
                   strides=conv_stride_size[3], padding='same', name='block3_conv1')(model)
    model = Activation('relu', name='block3_act1')(model)
    model = Conv1D(filters=filter_num[3], kernel_size=kernel_size[3],
                   strides=conv_stride_size[3], padding='same', name='block3_conv2')(model)
    model = Activation('relu', name='block3_act2')(model)
    model = MaxPooling1D(pool_size=pool_size[3], strides=pool_stride_size[3],
                         padding='same', name='block3_pool')(model)
    model = Dropout(0.1, name='block3_dropout')(model)

    model = Conv1D(filters=filter_num[4], kernel_size=kernel_size[4],
                   strides=conv_stride_size[4], padding='same', name='block4_conv1')(model)
    model = Activation('relu', name='block4_act1')(model)
    model = Conv1D(filters=filter_num[4], kernel_size=kernel_size[4],
                   strides=conv_stride_size[4], padding='same', name='block4_conv2')(model)
    model = Activation('relu', name='block4_act2')(model)
    model = MaxPooling1D(pool_size=pool_size[4], strides=pool_stride_size[4],
                         padding='same', name='block4_pool')(model)

    output = Flatten()(model)

    dense_layer = Dense(emb_size, name='FeaturesVec')(output)
    shared_conv2 = Model(inputs=input_data, outputs=dense_layer)
    return shared_conv2
