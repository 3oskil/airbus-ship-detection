from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping


def train(model, train_data, valid_data, epochs, model_path):
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
                        steps_per_epoch=1,
                        validation_steps=1,
                        callbacks=callbacks,
                        validation_data=valid_data)

    return history
