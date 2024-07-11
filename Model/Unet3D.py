import logging

from keras_unet_collection.activations import GELU, Snake
from tensorflow.keras import models
from tensorflow.keras.layers import *

from ConfigurationFiles.ConfigurationUtils import MODEL_CFG, get_configurations
from Process.ProcessUtils import get_filters
from Utils.CommonUtils import str_to_tuple

CLASS_NAME = "[Model/Unet3D]"
act = 'leakyRelu'


def init_activation(activation):
    global act
    lgr = CLASS_NAME + "[init_activation()]"
    if activation != "" and len(activation) > 1:
        act = activation
    else:
        logging.error(f"{lgr} No valid activation function is defined hence using LeakyReLU as "
                      f"default activation function.")


def Activation(x):
    lgr = CLASS_NAME + "[Activation()]"
    if act == 'leakyrelu':
        return LeakyReLU()(x)
    elif act == 'relu':
        return ReLU()(x)
    elif act == 'prelu':
        return PReLU()(x)
    elif act == 'gelu':
        return GELU()(x)
    elif act == 'snake':
        return Snake()(x)
    else:
        logging.error(f"{lgr} Invalid value given as activation function, using leakyReLU as default value.")
        return LeakyReLU()(x)


# 3D-Unet Down Sampling
def down_block(x, filters, use_maxpool=True):
    x = Conv3D(filters, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation(x)
    x = Conv3D(filters, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation(x)
    if use_maxpool:
        return MaxPooling3D(strides=(2, 2, 2), padding='same')(x), x
    else:
        return x


# 3D-Unet Up Sampling
def up_block(x, y, filters, use_transpose=True):
    if use_transpose:
        # Does up-sampling using learnable parameters. Adaptable
        x = Conv3DTranspose(filters, kernel_size=2, strides=2, padding='same')(x)
    else:
        # Does up-sampling by applying different mathematical functions. Fixed algorithm.
        x = UpSampling3D()(x)

    # For input images that are not in power of 2, the upsampling may cause a difference in shape. Adding cropping to fix this.
    if x.shape[1:-1] != y.shape[1:-1]:
        # Calculate the amount of cropping needed for each dimension
        crop_depth = max(0, x.shape[1] - y.shape[1])
        crop_height = max(0, x.shape[2] - y.shape[2])
        crop_width = max(0, x.shape[3] - y.shape[3])

        # Apply cropping
        x = Cropping3D(cropping=((0, crop_depth), (0, crop_height), (0, crop_width)))(x)

    x = Concatenate(axis=-1)([x, y])
    x = Conv3D(filters, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation(x)
    x = Conv3D(filters, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation(x)
    return x


class Unet3D:
    def __init__(self, input_shape, activation):
        self.input_shape = (*str_to_tuple(input_shape), 1)

        cfg = get_configurations(MODEL_CFG)

        self.output_classes = cfg["output_classes"]
        self.dropout = cfg["dropout"]
        self.filters = get_filters(cfg["min_filter"], 5)
        init_activation(activation.lower())
        self.print_info()

    def generate_model(self):
        img = Input(shape=self.input_shape)
        x, temp1 = down_block(img, self.filters[0])
        x, temp2 = down_block(x, self.filters[1])
        x, temp3 = down_block(x, self.filters[2])
        x, temp4 = down_block(x, self.filters[3])
        x = down_block(x, self.filters[4], use_maxpool=False)

        # decode
        x = up_block(x, temp4, self.filters[4])
        x = up_block(x, temp3, self.filters[2])
        x = up_block(x, temp2, self.filters[1])
        x = up_block(x, temp1, self.filters[0])

        x = Dropout(self.dropout)(x)

        # Update the output layer based on the number of classes
        if self.output_classes == 1:
            output = Conv3D(1, 1, activation='sigmoid')(x)
        else:
            output = Conv3D(self.output_classes, 1, activation='softmax')(x)

        out = models.Model(img, output, name='unet_3d')
        out.summary()
        return out

    def print_info(self):
        logging.info(f"{CLASS_NAME}[print_info()] Model initialized using the following configurations.")
        logging.info(self.__dict__)
