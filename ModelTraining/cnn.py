# import the necessary packages
from keras import backend as K, Model
from keras.applications import MobileNet
from keras.layers import GlobalAveragePooling2D, BatchNormalization, regularizers
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation, Dropout
from keras.layers.core import Dense
from keras.layers.core import Flatten
from keras.models import Sequential
from keras_applications.inception_v3 import InceptionV3
from keras_applications.mobilenet_v2 import MobileNetV2
from keras_applications.vgg19 import VGG19


class Cnn:
    @staticmethod
    def build_custom(width, height, depth, classes):
        # initialize the model
        model = Sequential()
        inputShape = (height, width, depth)

        # if we are using "channels first", update the input shape
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)

        # first set of CONV => RELU => POOL layers
        model.add(Conv2D(20, (5, 5), padding="same",
                         input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Dropout(0.2))

        # second set of CONV => RELU => POOL layers
        model.add(Conv2D(50, (5, 5), padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Dropout(0.2))


        # first (and only) set of FC => RELU layers
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation("relu"))
        model.add(Dropout(0.5))


        # softmax classifier
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        # return the constructed network architecture
        return model

    @staticmethod
    def build_transfer_learning(width, height, depth, classes, type):
        # create the base pre-trained model
        if type == 'VGG19':
            base_model = VGG19(input_shape=(height, width, depth), weights='imagenet',
                               include_top=False)
        elif type == 'MobileNet':
            base_model = MobileNet(input_shape=(height, width, depth), weights='imagenet',
                                   include_top=False, classes=classes)

        elif type == 'InceptionV3':
            base_model = InceptionV3(input_shape=(height, width, depth), weights='imagenet',
                                     include_top=False, classes=classes)
        elif type == 'MobileNetV2':
            base_model = MobileNetV2(input_shape=(height, width, depth), weights='imagenet',
                                     include_top=False, classes=classes)
        # add a global spatial average pooling layer
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.8)(x)
        # let's add a fully-connected layer
        x = Dense(1024)(x)
        x = Activation(activation='relu')(x)
        x = Dropout(0.8)(x)
        # and a logistic layer
        predictions = Dense(classes, activation='softmax')(x)

        # this is the model we will train
        model = Model(inputs=base_model.input, outputs=predictions)

        # train only the fully connected layers (which were randomly initialized)
        # i.e. freeze all convolutional layers
        for layer in base_model.layers:
            layer.trainable = False

        return model
