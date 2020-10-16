# import neccesary module

from keras.layers import Conv2D, BatchNormalization, Activation, Add


def conv_block(X, f, filters, stage, block, s=2):
    """
    For the convolution block the skip X size is diffenrent from the X
    before the non linear activation. that is why we need to pass it through
    conv layer. That is the only difference from the identity blocks
    So, Here the task is to build the ff architecture:

    X --> Linear--> relu --> Linear--> relu -->
    linear >> plus Linear(skip_X) --> relu

    """

    # unzip the filters
    filter1, filter2, filter3 = filters

    # define base name for conv and batck narm components
    conv_name = 'res_{}{}_block'.format(stage, block)
    bn_name = 'bn_{}{}_block'.format(stage, block)

    # set the X as skip_X for later use
    skip_X = X

    # block one
    X = Conv2D(filters=filter1, kernel_size=(1, 1), strides=(
        s, s), padding='valid', name=conv_name+'2a')(X)
    X = BatchNormalization(axis=3, name=bn_name+'2a')(X)
    X = Activation('relu')(X)

    # block two
    X = Conv2D(filters=filter2, kernel_size=(f, f), strides=(
        1, 1), padding='same', name=conv_name+'2b')(X)
    X = BatchNormalization(axis=3, name=bn_name+'2b')(X)
    X = Activation('relu')(X)

    # block three
    X = Conv2D(filters=filter3, kernel_size=(1, 1), strides=(
        1, 1), padding='valid', name=conv_name+'2c')(X)
    X = BatchNormalization(axis=3, name=bn_name+'2c')(X)

    # reduce the dimension
    skip_X = Conv2D(filters=filter3, kernel_size=(1, 1), strides=(s, s),
                    padding='valid', name=conv_name+'skip')(skip_X)
    skip_X = BatchNormalization(axis=3, name=bn_name+'skip')(skip_X)

    # add X and skip_X
    X = Add()([X, skip_X])

    # pass the sum through non linear function and return
    X = Activation('relu')(X)
    return X
