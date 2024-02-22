from utils.metrics import dice_score, bce_dice, true_positive_rate

from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.layers import Input, Conv2D, MaxPool2D, Conv2DTranspose, Dropout, BatchNormalization, concatenate


def build_unet_v1(config):
    img_size = config['img_size']

    strides = (2, 2)
    padding = 'same'
    pool_size = (2, 2)
    activation = 'relu'
    kernel_size_conv = (3, 3)
    kernel_size_conv_trans = (2, 2)
    kernel_initializer = 'he_normal'
    input_shape = (img_size, img_size, 3)
    learning_rate = config['learning_rate']

    inputs = Input(input_shape)

    c1 = Conv2D(16, kernel_size_conv, activation=activation,
                kernel_initializer=kernel_initializer, padding=padding)(inputs)

    c1 = Dropout(0.1)(c1)
    c1 = Conv2D(16, kernel_size_conv, activation=activation,
                kernel_initializer=kernel_initializer, padding=padding)(c1)
    p1 = MaxPool2D(pool_size)(c1)

    c2 = Conv2D(32, kernel_size_conv, activation=activation,
                kernel_initializer=kernel_initializer, padding=padding)(p1)
    c2 = Dropout(0.1)(c2)
    c2 = Conv2D(32, kernel_size_conv, activation=activation,
                kernel_initializer=kernel_initializer, padding=padding)(c2)
    p2 = MaxPool2D(pool_size)(c2)

    c3 = Conv2D(64, kernel_size_conv, activation=activation,
                kernel_initializer=kernel_initializer, padding=padding)(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(64, kernel_size_conv, activation=activation,
                kernel_initializer=kernel_initializer, padding=padding)(c3)
    p3 = MaxPool2D(pool_size)(c3)

    c4 = Conv2D(128, kernel_size_conv, activation=activation,
                kernel_initializer=kernel_initializer, padding=padding)(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(128, kernel_size_conv, activation=activation,
                kernel_initializer=kernel_initializer, padding=padding)(c4)
    p4 = MaxPool2D(pool_size)(c4)

    c5 = Conv2D(256, kernel_size_conv, activation=activation,
                kernel_initializer=kernel_initializer, padding=padding)(p4)
    c5 = Dropout(0.3)(c5)
    c5 = Conv2D(256, kernel_size_conv, activation=activation,
                kernel_initializer=kernel_initializer, padding=padding)(c5)

    u6 = Conv2DTranspose(128, kernel_size_conv_trans, strides=strides, padding=padding)(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, kernel_size_conv, activation=activation,
                kernel_initializer=kernel_initializer, padding=padding)(u6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv2D(128, kernel_size_conv, activation=activation,
                kernel_initializer=kernel_initializer, padding=padding)(c6)

    u7 = Conv2DTranspose(64, kernel_size_conv_trans, strides=strides, padding=padding)(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, kernel_size_conv, activation=activation,
                kernel_initializer=kernel_initializer, padding=padding)(u7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv2D(64, kernel_size_conv, activation=activation,
                kernel_initializer=kernel_initializer, padding=padding)(c7)

    u8 = Conv2DTranspose(32, kernel_size_conv_trans, strides=strides, padding=padding)(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, kernel_size_conv, activation=activation,
                kernel_initializer=kernel_initializer, padding=padding)(u8)
    c8 = Dropout(0.1)(c8)
    c8 = Conv2D(32, kernel_size_conv, activation=activation,
                kernel_initializer=kernel_initializer, padding=padding)(c8)

    u9 = Conv2DTranspose(16, kernel_size_conv_trans, strides=strides, padding=padding)(c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(16, kernel_size_conv, activation=activation,
                kernel_initializer=kernel_initializer, padding=padding)(u9)
    c9 = Dropout(0.1)(c9)
    c9 = Conv2D(16, kernel_size_conv, activation=activation,
                kernel_initializer=kernel_initializer, padding=padding)(c9)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)

    optimizer = Adam(learning_rate=learning_rate)
    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer=optimizer, loss=bce_dice, metrics=[dice_score, true_positive_rate])

    return model


def build_unet_v2(config):
    k = 2
    img_size = config['img_size']
    input_shape = (img_size, img_size, 3)
    learning_rate = config['learning_rate']

    def conv_block(x_, filters_, kernel_size=3, activation="elu", initializer='he_normal', dropout=0.2):
        x_ = BatchNormalization()(x_)
        x_ = Conv2D(filters_, kernel_size, padding="same", activation=activation, kernel_initializer=initializer)(x_)
        x_ = Dropout(dropout)(x_)
        x_ = Conv2D(filters_, kernel_size, padding="same", activation=activation, kernel_initializer=initializer)(x_)
        return x_

    inputs = Input(shape=input_shape)
    skips = []
    x = inputs

    # Downward path
    for filters in [k * 16, k * 32, k * 64, k * 128]:
        x = conv_block(x, filters)
        skips.append(x)
        x = MaxPool2D(strides=2)(x)

    # Bridge
    x = conv_block(x, k * 256)

    # Upward path
    for filters, skip in zip([k * 128, k * 64, k * 32, k * 16], reversed(skips)):
        x = Conv2DTranspose(filters // 2, 2, strides=2)(x)
        x = concatenate([x, skip])
        x = conv_block(x, filters)

    # Final layer
    outputs = Conv2D(filters=1, kernel_size=1, padding='same', activation='sigmoid')(x)

    optimizer = Adam(learning_rate=learning_rate)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer, loss=bce_dice, metrics=[dice_score, true_positive_rate])

    return model


def build_unet_pp(config):
    img_size = config['img_size']
    nb_filter = config['nb_filter']
    learning_rate = config['learning_rate']

    dropout = 0.5
    padding = 'same'
    activation = 'elu'
    strides_pool = (2, 2)
    kernel_size_pool = (2, 2)
    kernel_size_conv = (3, 3)
    strides_conv_trans = (2, 2)
    kernel_regularizer = l2(0.001)
    kernel_size_conv_trans = (2, 2)
    kernel_initializer = 'he_normal'
    input_shape = (img_size, img_size, 3)

    inputs = Input(input_shape)

    def conv2d_block(inputs_, filters):
        c = Conv2D(filters, kernel_size_conv, padding=padding, activation=activation,
                   kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)(inputs_)
        c = Dropout(dropout)(c)
        c = Conv2D(filters, kernel_size_conv, padding=padding, activation=activation,
                   kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)(c)
        c = Dropout(dropout)(c)
        return c

    c1 = conv2d_block(inputs, 32)
    pool1 = MaxPool2D(kernel_size_pool, strides_pool)(c1)

    c2 = conv2d_block(pool1, 64)
    pool2 = MaxPool2D(kernel_size_pool, strides_pool)(c2)

    up1_2 = Conv2DTranspose(nb_filter[0], kernel_size_conv_trans, strides_conv_trans, name='up12', padding='same')(c2)
    conv1_2 = concatenate([up1_2, c1], name='merge12', axis=3)
    c3 = conv2d_block(conv1_2, 32)

    conv3_1 = conv2d_block(pool2, 128)
    pool3 = MaxPool2D(kernel_size_pool, strides_pool, name='pool3')(conv3_1)

    up2_2 = Conv2DTranspose(nb_filter[1], kernel_size_conv_trans, strides_conv_trans, name='up22', padding='same')(
        conv3_1)
    conv2_2 = concatenate([up2_2, c2], name='merge22', axis=3)  # x10
    conv2_2 = conv2d_block(conv2_2, 64)

    up1_3 = Conv2DTranspose(nb_filter[0], kernel_size_conv_trans, strides_conv_trans, name='up13', padding='same')(
        conv2_2)
    conv1_3 = concatenate([up1_3, c1, c3], name='merge13', axis=3)
    conv1_3 = conv2d_block(conv1_3, 32)

    conv4_1 = conv2d_block(pool3, 256)
    pool4 = MaxPool2D(kernel_size_pool, strides_pool, name='pool4')(conv4_1)

    up3_2 = Conv2DTranspose(nb_filter[2], kernel_size_conv_trans, strides_conv_trans, name='up32', padding='same')(
        conv4_1)
    conv3_2 = concatenate([up3_2, conv3_1], name='merge32', axis=3)  # x20
    conv3_2 = conv2d_block(conv3_2, 128)

    up2_3 = Conv2DTranspose(nb_filter[1], kernel_size_conv_trans, strides_conv_trans, name='up23', padding='same')(
        conv3_2)
    conv2_3 = concatenate([up2_3, c2, conv2_2], name='merge23', axis=3)
    conv2_3 = conv2d_block(conv2_3, 64)

    up1_4 = Conv2DTranspose(nb_filter[0], kernel_size_conv_trans, strides_conv_trans, name='up14', padding='same')(
        conv2_3)
    conv1_4 = concatenate([up1_4, c1, c3, conv1_3], name='merge14', axis=3)
    conv1_4 = conv2d_block(conv1_4, 32)

    conv5_1 = conv2d_block(pool4, 512)

    up4_2 = Conv2DTranspose(nb_filter[3], kernel_size_conv_trans, strides_conv_trans, name='up42', padding='same')(
        conv5_1)
    conv4_2 = concatenate([up4_2, conv4_1], name='merge42', axis=3)  # x30
    conv4_2 = conv2d_block(conv4_2, 256)

    up3_3 = Conv2DTranspose(nb_filter[2], kernel_size_conv_trans, strides_conv_trans, name='up33', padding='same')(
        conv4_2)
    conv3_3 = concatenate([up3_3, conv3_1, conv3_2], name='merge33', axis=3)
    conv3_3 = conv2d_block(conv3_3, 128)

    up2_4 = Conv2DTranspose(nb_filter[1], kernel_size_conv_trans, strides_conv_trans, name='up24', padding='same')(
        conv3_3)
    conv2_4 = concatenate([up2_4, c2, conv2_2, conv2_3], name='merge24', axis=3)
    conv2_4 = conv2d_block(conv2_4, 64)

    up1_5 = Conv2DTranspose(nb_filter[0], kernel_size_conv_trans, strides_conv_trans, name='up15', padding='same')(
        conv2_4)
    conv1_5 = concatenate([up1_5, c1, c3, conv1_3, conv1_4], name='merge15', axis=3)
    conv1_5 = conv2d_block(conv1_5, 32)

    output_4 = Conv2D(1, (1, 1), activation='sigmoid', kernel_initializer='he_normal',
                      name='output_4', padding='same')(conv1_5)

    model = Model([inputs], [output_4])

    model.compile(optimizer=Adam(learning_rate=learning_rate), loss=bce_dice, metrics=[dice_score, true_positive_rate])

    return model
