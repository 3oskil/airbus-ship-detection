import keras.backend as K


def dice_score(y_true, y_pred):
    return (2.0 * K.sum(y_pred * y_true) + 0.0001) / (K.sum(y_true) + K.sum(y_pred) + 0.0001)


def bce_dice(y_true, y_pred):
    return K.binary_crossentropy(y_true, y_pred) + (1 - dice_score(y_true, y_pred))


def true_positive_rate(y_true, y_pred):
    return K.sum(K.flatten(y_true) * K.flatten(K.round(y_pred))) / K.sum(y_true)
