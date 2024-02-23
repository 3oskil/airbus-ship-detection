from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping


def train(model, train_data, valid_data, epochs, model_path):
    """
    Trains a model with given training and validation data, employing callbacks for early stopping,
    learning rate reduction, and model checkpointing to save the best model based on validation loss.

    Parameters:
        model (keras.Model): The model to be trained.
        train_data (keras.utils.Sequence or tf.data.Dataset): The training data.
        valid_data (keras.utils.Sequence or tf.data.Dataset): The validation data used for evaluating the model.
        epochs (int): The number of epochs to train the model.
        model_path (str): The file path where the best model weights should be saved.

    Returns:
        keras.callbacks.History: A history object containing the training history metrics.
    """
    checkpoint = ModelCheckpoint(model_path,
                                 monitor='val_loss',
                                 verbose=1,
                                 save_best_only=True,
                                 save_weights_only=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                  factor=0.5,
                                  patience=3,
                                  verbose=1,
                                  min_delta=0.0001,
                                  min_lr=1e-6)
    early_stop = EarlyStopping(monitor='val_loss',
                               patience=5,
                               verbose=1)

    callbacks = [checkpoint, reduce_lr, early_stop]

    history = model.fit(train_data,
                        epochs=epochs,
                        verbose=1,
                        callbacks=callbacks,
                        validation_data=valid_data)

    return history
