import tensorflow as tf
from tensorflow.keras.losses import BinaryCrossentropy

# -- Constants -- #
CUSTOM_PERF_METRICS = "dice_coef"
CUSTOM_LOSS_FUNCTIONS = "dice,tverskycoef,tversky,focal_tversky,bce_dice"
CUSTOM_ACTIVATIONS = "GELU,Snake"


# -- Performance Metrics -- #

def dice_coef(y_true, y_pred, smooth=1e-4):
    # First axis is added for ensure batches are considered for dice calculation.
    # Assuming tensor is channel last. Expected Tensor Shape (B, H, W, D, C)
    axis = list(range(0, len(y_true.shape) - 1))

    # y_true_pos = tf.keras.backend.flatten(y_true)
    # y_pred_pos = tf.keras.backend.flatten(y_pred)
    true_pos = tf.reduce_sum((y_true * y_pred), axis=axis, keepdims=False)  # Sum across spatial dimensions
    false_neg = tf.reduce_sum((y_true * (1 - y_pred)), axis=axis, keepdims=False)
    false_pos = tf.reduce_sum(((1 - y_true) * y_pred), axis=axis, keepdims=False)

    n = 2.0 * true_pos + smooth
    d = 2.0 * true_pos + false_pos + false_neg + smooth
    # Need to check clip_by_value function

    return n / tf.clip_by_value(d, d, 1e-7)


# -- Loss Functions -- #
@tf.function
def dice(y_true, y_pred, smooth=1e-7):
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)

    return tf.reduce_mean(1 - dice_coef(y_true, y_pred, smooth))


@tf.function
def bce_dice(y_true, y_pred):
    bce = BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
    return tf.reduce_mean(bce(y_true, y_pred) + dice(y_true, y_pred))
