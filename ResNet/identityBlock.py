# import neccesary module
from keras.layers import Conv2D, BatchNormalization, Activation, Add


def identity(X, f, filters, stage, block):
    """
    For the identity block the X from the skip connection is
    similar with the output of the block
    Here what we should implement is:
    X --> Linear --> activation --> Linear--> activation --> Linear -->
              plus X --> activation
    """

    filter1, filter2, filter3 = filters
    # defining names for BatchNormalization & Convolution blocks in the network
    conv_name = 'res_{}{}_block'.format(stage, block)
    bn_name = 'bn_{}{}_block'.format(stage, block)
    # shortcut/ skip X
    skip_X = X
    # block one
    X = Conv2D(filters=filter1, kernel_size=(1, 1), strides=(1, 1),
               padding='valid', name=conv_name+'2a')(X)

    X = BatchNormalization(axis=3, name=bn_name+'2a')(X)
    X = Activation('relu')(X)

    # block two
    X = Conv2D(filters=filter2, kernel_size=(f, f), strides=(1, 1),
               padding='same', name=conv_name+'2b')(X)
    X = BatchNormalization(axis=3, name=bn_name+'2b')(X)
    X = Activation('relu')(X)

    # block three
    X = Conv2D(filters=filter3, kernel_size=(1, 1), strides=(
        1, 1), padding='valid', name=conv_name+'2c')(X)
    X = BatchNormalization(axis=3, name=bn_name+'2c')(X)

    # add the shortcut with X before activation and pass it through the
    X = Add()([X, skip_X])
    X = Activation('relu')(X)
    # finally return the identity block output
    return X
