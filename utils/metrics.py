import keras.backend as K


def dice_score(y_true, y_pred):
    """
    Calculates the Dice coefficient, a measure of set similarity, which is especially useful for evaluating image
    segmentation models.

    Parameters:
        y_true (tensor): The ground truth tensor, same shape as y_pred.
        y_pred (tensor): The predicted tensor, same shape as y_true.

    Returns:
        float: Dice coefficient between the predicted and ground truth tensors.
    """
    # Calculate the numerator and denominator of the Dice coefficient, adding a small constant to avoid division by zero
    numerator = 2.0 * K.sum(y_pred * y_true) + 0.0001
    denominator = K.sum(y_true) + K.sum(y_pred) + 0.0001
    # Return the Dice coefficient
    return numerator / denominator


def bce_dice(y_true, y_pred):
    """
    Computes the sum of binary cross-entropy loss and the Dice loss (1 - Dice coefficient).
    This combined loss is useful for balancing the contribution of the pixel-wise binary cross-entropy loss
    and the global Dice coefficient, improving segmentation model performance.

    Parameters:
        y_true (tensor): The ground truth tensor, same shape as y_pred.
        y_pred (tensor): The predicted tensor, same shape as y_true.

    Returns:
        tensor: The sum of binary cross-entropy and (1 - Dice score).
    """
    # Calculate binary cross-entropy loss
    bce_loss = K.binary_crossentropy(y_true, y_pred)
    # Calculate Dice loss as 1 minus the Dice score
    dice_loss = 1 - dice_score(y_true, y_pred)
    # Return the sum of the BCE loss and Dice loss
    return bce_loss + dice_loss


def true_positive_rate(y_true, y_pred):
    """
    Calculates the true positive rate (TPR), or sensitivity, measuring the proportion of actual positives that are
    correctly identified.

    Parameters:
        y_true (tensor): The ground truth tensor, same shape as y_pred.
        y_pred (tensor): The predicted tensor, same shape as y_true.

    Returns:
        float: The true positive rate of the predictions.
    """
    # Flatten the tensors and calculate the true positives by multiplying the ground truth by the rounded predictions
    true_positives = K.sum(K.flatten(y_true) * K.flatten(K.round(y_pred)))
    # Calculate the total number of actual positives
    actual_positives = K.sum(y_true)
    # Return the true positive rate
    return true_positives / actual_positives
