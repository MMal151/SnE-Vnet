import logging
import os

import nibabel as nib
import numpy as np
import tensorflow as tf
import SimpleITK as sitk
from keras_unet_collection.activations import GELU, Snake
from tensorflow.keras.metrics import *
from tensorflow.keras.optimizers import *
from keras_unet_collection.losses import tversky_coef, tversky, focal_tversky

from ConfigurationFiles.ConfigurationUtils import LR_CFG, get_configurations
from Misc.CarveMix import generate_new_sample
from Utils.CommonUtils import get_random_index, is_valid_file, is_valid_str
from Utils.CustomFunctions import CUSTOM_ACTIVATIONS, CUSTOM_LOSS_FUNCTIONS, bce_dice, dice, dice_coef

CLASS_NAME = "[Process/ProcessUtils]"


def load_model(cfg, file):
    model = None
    if is_valid_file(file):
        model = tf.keras.models.load_model(file,
                                           custom_objects=get_custom_objects(cfg))

    return model


# Custom Objects includes all custom activations, loss functions, and performance metrics.
def get_custom_objects(cfg):
    lgr = CLASS_NAME + "[get_custom_objects()]"
    custom_objects = get_metrics(cfg["perf_metrics"])

    if CUSTOM_ACTIVATIONS.__contains__(cfg["activation"]):
        logging.debug(f"{lgr}: Adding custom activation function in custom_obj list.")
        custom_objects[cfg["activation"]] = get_activation(cfg["activation"])

    if CUSTOM_LOSS_FUNCTIONS.__contains__(cfg["loss"].lower()):
        logging.debug(f"{lgr}: Adding custom loss function in custom_obj list.")
        custom_objects[cfg["loss"]] = get_loss(cfg)

        # if cfg["train"]["loss"].lower() == "focal_tversky":
        #   custom_objects["focal_tversky"] = loss

    return custom_objects


def get_metrics(met_list="dice_coef"):
    lgr = CLASS_NAME + "[get_metrics_test()]"
    metrics = {}

    if is_valid_str(met_list):
        if met_list.__contains__("acc"):
            metrics["acc"] = Accuracy(name="acc")
        if met_list.__contains__("mean_iou"):
            metrics["mean_iou"] = MeanIoU(num_classes=2, name="mean_iou")
        if met_list.__contains__("recall"):
            metrics["recall"] = Recall(name="recall")
        if met_list.__contains__("prec"):
            metrics["prec"] = Precision(name="prec")
        if met_list.__contains__("dice_coef"):
            metrics["dice_coef"] = dice_coef
    else:
        logging.info(f"{lgr}: Invalid input string: [{met_list}]. Using Dice Coefficient as the default metric.")
        metrics["dice_coef"] = dice_coef

    return metrics


def get_loss(loss):
    lgr = CLASS_NAME + "[get_loss()]"

    if loss == 'dice':
        logging.debug(f"{lgr}: Using Dice Function as Loss Function.")
        return dice
    elif loss == 'tversky_coef':
        logging.debug(f"{lgr}: Using Tversky Coef as Loss Function.")
        return tversky_coef
    elif loss == "tversky":
        logging.debug(f"{lgr}: Using Tversky as Loss Function.")
        return tversky
    elif loss == "focal_tversky":
        logging.debug(f"{lgr}: Using Focal Tversky as Loss Function.")
        return focal_tversky
    elif loss == 'bce_dice':
        logging.debug(f"{lgr}: Using sum of binary cross-entropy and dice as Loss Function.")
        return bce_dice

    logging.debug(f"{lgr}: Using {loss} as Loss Function.")
    return loss


def get_activation(act):
    if act == 'GELU':
        return GELU()
    elif act == 'Snake':
        return Snake()

    return None


def get_schedular(lr):
    # Reading Learning Rate Schedular configurations from config file.
    cfg = get_configurations(LR_CFG)

    lgr = CLASS_NAME + "[get_schedular()]"
    opt = cfg["type"].lower()

    if opt == "cosine":
        decay_steps = 1000

        if cfg["decay_steps"] > 0:
            decay_steps = cfg["decay_steps"]

        logging.debug(f"{lgr}: Using Cosine Decay learning schedular with configurations initial_learning_rate: [{lr}]"
                      f" decay_steps: [{decay_steps}]")

        return tf.keras.optimizers.schedules.CosineDecay(lr, decay_steps)

    elif opt == "exponential":
        decay_steps = 100000
        decay_rate = 1000

        if cfg["decay_steps"] > 0:
            decay_steps = cfg["decay_steps"]
        if cfg["exponential"]["decay_rate"] > -1:
            decay_rate = cfg["exponential"]["decay_rate"]

        logging.debug(f"{lgr}: Using Exponential learning schedular with configurations initial_learning_rate: [{lr}]"
                      f" decay_steps: [{decay_steps}]"
                      f" and decay_rate: [{decay_rate}].")

        return tf.keras.optimizers.schedules.ExponentialDecay(lr, decay_steps, decay_rate)

    elif opt == "polynomial":
        decay_steps = 10000
        end_learning_rate = 0.01

        if cfg["decay_steps"] > 0:
            decay_steps = cfg["decay_steps"]
        if cfg["polynomial"]["end_learning_rate"] > -1:
            end_learning_rate = cfg["polynomial"]["end_learning_rate"]

        logging.debug(f"{lgr}: Using Polynomial learning schedular with configurations initial_learning_rate: [{lr}]"
                      f" decay_steps: [{decay_steps}]"
                      f" and end_learning_rate: [{end_learning_rate}].")

        return tf.keras.optimizers.schedules.PolynomialDecay(lr, decay_steps, end_learning_rate)

    return lr


def get_optimizer(cfg):
    lgr = CLASS_NAME + "[get_optimizer()]"

    opt, lr = cfg["optimizer"].lower(), cfg["learning_rate"]

    if cfg["aply_lr_sch"]:
        lr = get_schedular(lr)

    if opt == 'adam':
        return Adam(learning_rate=lr)
    elif opt == 'sgd':
        return SGD(learning_rate=lr)
    elif opt == 'rmsprop':
        return RMSprop(learning_rate=lr)
    elif opt == 'adadelta':
        return Adadelta(learning_rate=lr)
    elif opt == 'adamax':
        return Adamax(learning_rate=lr)
    elif opt == 'adagrad':
        return Adagrad(learnitng_rate=lr)
    else:
        logging.error(f"{lgr}: Invalid Optimizer, using Adam as default optimizer.")
        return Adam(learning_rate=lr)


# Generate list of filters using configured filter value.
def get_filters(min_filter, tot_filters):
    lgr = CLASS_NAME + "[get_filters()]"

    filters = [8, 16, 32, 64, 128]
    if tot_filters > 0:
        filters.clear()
        curr_filter = min_filter
        for i in range(0, tot_filters):
            filters.append(curr_filter)
            curr_filter *= 2
    else:
        logging.info(f"{lgr}: Invalid value of total number of filters. Returning default value [8, 16, 32, 64, 128]")
    return filters


# -- Pre-processing Utils --#
# Source: https://github.com/fitushar/3D-Medical-Imaging-Preprocessing-All-you-need
def normalize_img(image, smooth=1e-8):
    mean = np.mean(image)
    std = np.std(image)
    image -= mean
    image /= (max(std, smooth))
    return image


def augmentation_cm(x, y, x_ext, y_ext, factor):
    lgr = CLASS_NAME + "[augmentation_cm()]"

    # Total number of datapoints to be augmented.
    total_aug_dp = int(np.floor(len(x) * factor))

    # Storing the state of len before adding augmented datapoints.
    # This ensures that no augmented datapoint will be used for further augmentation.
    datapoints = len(x)

    if total_aug_dp < datapoints:
        logging.info(f"{lgr}: CraveMix-based Augmentation started. ")

        for k in range(0, total_aug_dp):
            i, j = get_random_index(0, datapoints - 1)
            logging.debug(f"{lgr}: Augmenting Datapoints {x[i]} and {x[j]}")
            vol_a, vol_b = nib.imagestats.mask_volume(nib.load(y[i])), nib.imagestats.mask_volume(nib.load(y[j]))

            # This comparison is missing from the original cravemix algorithm. Having this comparison however ensures
            # smoother augmentation. The image given as input first acts as the base image. If the bigger lesion is
            # augmented in the image imbalances the intensities of the base image.
            if vol_a > vol_b:
                new_img, new_label, _, _ = generate_new_sample(x[i], x[j], y[i], y[j])
            else:
                new_img, new_label, _, _ = generate_new_sample(x[j], x[i], y[j], y[i])

            new_file = x[i].split("/")[-2] + "_" + x[j].split("/")[-2] + "_cm"
            in_path = x[i].split("sub")[0]
            path_new = os.path.join(in_path, new_file)
            os.makedirs(path_new, exist_ok=True)

            img_path = os.path.join(path_new, new_file + "_" + x_ext)
            lbl_path = os.path.join(path_new, new_file + "_" + y_ext)

            sitk.WriteImage(new_img, img_path)
            sitk.WriteImage(new_label, lbl_path)
            x.append(img_path)
            y.append(lbl_path)

        logging.info(f"{lgr}: CarveMix-based Augmentation completed. ")

    else:
        logging.error(f"{lgr}: Invalid augmentation factor. "
                      f"Total number of augmented datapoints should not exceed the total number of datapoints.")

    return x, y
