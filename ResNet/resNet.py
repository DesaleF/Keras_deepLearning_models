# ResNet50
# First thing first, import all the neccesary modules

import matplotlib.pyplot as plt
from keras.utils import plot_model
from keras.preprocessing import image
from keras.models import Model
from keras.layers import MaxPooling2D
from keras.layers import BatchNormalization, Flatten, Conv2D, AveragePooling2D
from keras.layers import Input, Dense, Activation, ZeroPadding2D
import numpy as np
import keras.backend as K
import os

from identityBlock import identity
from convolutionalBlock import conv_block

K.set_image_data_format('channels_last')
K.set_learning_phase(1)


def ResNet50(input_shape=(150, 150, 2), classes=2):
    '''
        if the input shape is different, just provide the shape during function
        call. the number of class may vary depending on the problem set
    '''
    # the input to the keras model
    X_input = Input(shape=input_shape)
    # the output to the keras model
    X = ZeroPadding2D(padding=(3, 3))(X_input)
    # now the first convolution using 7x7 filter
    X = Conv2D(64, kernel_size=(7, 7), strides=(1, 1), name='conv_1')(X)
    X = BatchNormalization(axis=3, name='bn1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(X)

    # The first block one conv_block two identity blocks
    X = conv_block(X, f=3, filters=[64, 64, 256], stage=2, block='a', s=1)
    X = identity(X, f=3, filters=[64, 64, 256], stage=2, block='b')
    X = identity(X, f=3, filters=[64, 64, 256], stage=2, block='c')

    # second block
    X = conv_block(X, f=3, filters=[128, 128, 512], stage=3, block='a', s=2)
    X = identity(X, f=3, filters=[128, 128, 512], stage=3, block='b')
    X = identity(X, f=3, filters=[128, 128, 512], stage=3, block='c')
    X = identity(X, f=3, filters=[128, 128, 512], stage=3, block='d')

    # third block
    X = conv_block(X, f=3, filters=[256, 256, 1024], stage=4, block='a', s=2)
    X = identity(X, f=3, filters=[256, 256, 1024], stage=4, block='b')
    X = identity(X, f=3, filters=[256, 256, 1024], stage=4, block='c')
    X = identity(X, f=3, filters=[256, 256, 1024], stage=4, block='d')
    X = identity(X, f=3, filters=[256, 256, 1024], stage=4, block='e')
    X = identity(X, f=3, filters=[256, 256, 1024], stage=4, block='f')

    # fourth block
    X = conv_block(X, f=3, filters=[512, 512, 2048], stage=5, block='a', s=2)
    X = identity(X, f=3, filters=[512, 512, 2048], stage=5, block='b')
    X = identity(X, f=3, filters=[512, 512, 2048], stage=5, block='c')

    X = AveragePooling2D(pool_size=(2, 2), name='average_pool')(X)

    # flatten X for the fully connected layer
    X = Flatten()(X)
    X = Dense(units=classes,
              activation='softmax',
              name='fc_{}'.format(classes)
              )(X)

    # finally cleate a model using X_input and X
    model = Model(inputs=X_input, name='ResNet50', outputs=X)
    return model


if __name__ == '__main__':
    print('no module error')

    # just to see the filenames one in each category
    for dirname, p, filenames in os.walk('intel_image_classification'):
        for filename in filenames:
            print(os.path.join(dirname, filename), p)
            break
    print('_________________________________________________________________')
    # rescale pixel values for efficiency , data augmentation for train set
    train_proces = image.ImageDataGenerator(rescale=1./255, zoom_range=0.2,
                                            horizontal_flip=True)
    # will not data augment for test set
    test_proces = image.ImageDataGenerator(rescale=1./255)

    train_set = train_proces.flow_from_directory(
        directory='intel_image_classification/seg_train/seg_train',
        batch_size=64, target_size=(150, 150))  # default hot one encoding
    test_set = test_proces.flow_from_directory(
        directory='intel_image_classification/seg_test/seg_test',
        batch_size=64, target_size=(150, 150))  # default hot one encoding

    # check the shape
    print('Train set dimension: {}'.format(train_set[0][0].shape))
    print('Test set dimension:  {}'.format(test_set[0][0].shape))

    # for labeling simplity
    classes = ["building", "forest", "glacier", "mountain", "sea", "street"]

    # plotting data, for just checking
    # number of images per row and number of rows
    # n should be less than 64
    n = 3
    # syntax for create subplots
    fig, axs = plt.subplots(n, n)
    fig.tight_layout()
    # next image after each iteration
    c = 1
    for i in range(n):
        for j in range(n):
            # selected a batch
            axs[i, j].imshow(np.reshape(train_set[0][0][c], [150, 150, 3]))
            axs[i, j].set_title(classes[np.argmax(train_set[0][1][c])])
            c += 1
    plt.show()

    # Now, create the model and train using the above data
    model = ResNet50(input_shape=(150, 150, 3), classes=6)
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy']
                  )
    model.summary()

    # train The model
    history = model.fit(train_set,
                        validation_data=test_set,
                        epochs=1,
                        verbose=1
                        )
    # save the trained model and save modelplot as image
    model.save('saved_model')
    plot_model(model, to_file='model.png')

    # evaluate the performance of the model
    preds = model.evaluate(test_set)
    print("Loss = " + str(preds[0]))
    print("Test Accuracy = " + str(preds[1]))
