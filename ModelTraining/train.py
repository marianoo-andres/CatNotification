import matplotlib

# set the matplotlib backend so figures can be saved in the background
matplotlib.use("Agg")
from keras.callbacks import TensorBoard, ModelCheckpoint
import random
from keras.optimizers import Adam
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array, load_img, ImageDataGenerator
import matplotlib.pyplot as plt
from cnn import Cnn
from keras.utils import plot_model
import os
import numpy as np
import tensorflow as tf
import random as rn
os.environ['PYTHONHASHSEED'] = '0'

# The below is necessary for starting Numpy generated random numbers
# in a well-defined initial state.

np.random.seed(42)

# The below is necessary for starting core Python generated random numbers
# in a well-defined state.

rn.seed(12345)

# Force TensorFlow to use single thread.
# Multiple threads are a potential source of
# non-reproducible results.
# For further details, see: https://stackoverflow.com/questions/42022950/which-seeds-have-to-be-set-where-to-realize-100-reproducibility-of-training-res

#session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

from keras import backend as K

# The below tf.set_random_seed() will make random number generation
# in the TensorFlow backend have a well-defined initial state.
# For further details, see: https://www.tensorflow.org/api_docs/python/tf/set_random_seed

tf.set_random_seed(1234)

#sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
#K.set_session(sess)

base_path = os.path.dirname(os.path.realpath(__file__))

def save_loss_plot(H, file_name):
    plt.figure(figsize=[8, 6])
    plt.plot(H.history['loss'], 'r', linewidth=2.0)
    plt.plot(H.history['val_loss'], 'b', linewidth=2.0)
    plt.legend(['Training loss', 'Validation Loss'], fontsize=18)
    plt.xlabel('Epochs ', fontsize=16)
    plt.ylabel('Loss', fontsize=16)
    plt.title('Loss Curves', fontsize=16)
    plt.savefig(file_name)


def save_accuracy_plot(H, file_name):
    plt.figure(figsize=[8, 6])
    plt.plot(H.history['acc'], 'r', linewidth=2.0)
    plt.plot(H.history['val_acc'], 'b', linewidth=2.0)
    plt.legend(['Training Accuracy', 'Validation Accuracy'], fontsize=18)
    plt.xlabel('Epochs ', fontsize=16)
    plt.ylabel('Accuracy', fontsize=16)
    plt.title('Accuracy Curves', fontsize=16)
    plt.savefig(file_name)


def train():
    # initialize the number of epochs to train for, initia learning rate,
    # and batch size
    RANDOM_SEED = 12345
    EPOCHS = 1000
    INIT_LR = 1e-3
    TRAINING_BATCH_SIZE = 32
    IMAGE_SIZE = 64
    NUM_CLASS = 2
    MODEL_TYPE = "MobileNetV2"
    TEST_SIZE = 0

    # initialize the data and labels
    print("[INFO] loading images...")
    inputs = []
    labels = []

    # grab the image paths and randomly shuffle them
    image_names = sorted(os.listdir(os.path.join(base_path, 'Data', 'Training')))
    image_paths = []
    for image_name in image_names:
        image_paths.append(os.path.join(base_path, 'Data', 'Training', image_name))
    random.shuffle(image_paths)

    # loop over the input images
    for image_path in image_paths:
        # load the image, pre-process it, and store it in the data list
        image = load_img(image_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
        array_image = img_to_array(image)
        inputs.append(array_image)

        # extract the class label from the image path and update the
        # labels list
        image_name = image_path.split(os.path.sep)[-1]
        type = image_name.split('_')[0]
        if type == 'gato':
            label = 1
        elif type == 'nogato':
            label = 0
        labels.append(label)

    # Convert labels to numpy
    labels = np.array(labels)

    # scale the raw pixel intensities to the range [0, 1]
    inputs = np.array(inputs, dtype="float") / 255.0

    """ADD TEST"""
    test_inputs = []
    test_labels = []
    # grab the image paths and randomly shuffle them
    test_image_names = sorted(os.listdir(os.path.join(base_path, 'Data', 'Validation')))
    test_image_paths = []
    for test_image_name in test_image_names:
        test_image_paths.append(os.path.join(base_path, 'Data', 'Validation', test_image_name))
    random.shuffle(image_paths)

    # loop over the input images
    for test_image_path in test_image_paths:
        if ".jpg" not in test_image_path:
            continue
        # load the image, pre-process it, and store it in the data list
        image = load_img(test_image_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
        array_image = img_to_array(image)
        test_inputs.append(array_image)

        # extract the class label from the image path and update the
        # labels list
        test_image_name = test_image_path.split(os.path.sep)[-1]
        type = test_image_name.split('_')[0]
        if type == 'gato':
            label = 1
        elif type == 'nogato':
            label = 0
        test_labels.append(label)

    # Convert labels to numpy
    test_labels = np.array(test_labels)
    test_labels = to_categorical(test_labels, num_classes=NUM_CLASS)
    # scale the raw pixel intensities to the range [0, 1]
    test_inputs = np.array(test_inputs, dtype="float") / 255.0
    """END TEST"""
    # partition the data into training and testing splits using 90% of
    # the data for training and the remaining 10% for testing
    (train_x, test_x, train_y, test_y) = train_test_split(inputs,
                                                          labels, test_size=TEST_SIZE,
                                                          random_state=RANDOM_SEED)
    # convert the labels from integers to vectors
    train_y = to_categorical(train_y, num_classes=NUM_CLASS)
    test_y = to_categorical(test_y, num_classes=NUM_CLASS)

    """TEST"""
    test_x = np.vstack((test_x, test_inputs))
    test_y = np.vstack((test_y, test_labels))
    """END TEST"""

    # # construct the image generator for data augmentation
    data_gen_augmentation = ImageDataGenerator(horizontal_flip=True)
    #data_gen_augmentation = ImageDataGenerator()

    """
    Decomment to preview images augmented
    
    data_gen_augmentation.fit(train_x)
    # the .flow() command below generates batches of randomly transformed images
    # and saves the results to the `preview/` directory
    i = 0
    for batch in data_gen_augmentation.flow(train_x, batch_size=1,
                                            save_to_dir='preview', save_prefix='cat',
                                            save_format='jpeg'):
        i += 1
        if i > 20:
            break  # otherwise the generator would loop indefinitely
    return
    
    """
    # initialize the model
    print("[INFO] compiling model...")
#    model = Cnn.build_transfer_learning(width=IMAGE_SIZE, height=IMAGE_SIZE, depth=3,
#                                        classes=NUM_CLASS, type=MODEL_TYPE)
    model = Cnn.build_custom(width=IMAGE_SIZE, height=IMAGE_SIZE, depth=3, classes=NUM_CLASS)
    plot_model(model, to_file='model_topology.png', show_shapes=True)
    # optimizer = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
    optimizer = Adam(lr=INIT_LR)
    model.compile(loss="binary_crossentropy", optimizer=optimizer,
                  metrics=["accuracy"])

    # Callbacks
    tensorboard = TensorBoard(log_dir='Graph', histogram_freq=0,
                              write_graph=True, write_images=True)
    file_name = "epoch_{epoch:02d}_valLoss_{val_loss:.6f}.h5"
    checkpoint = ModelCheckpoint(os.path.join("Models", "Training", file_name), monitor='val_loss',
                                 save_best_only=False)
    callbacks = [checkpoint, tensorboard]

    # train the network
    print("[INFO] training network...")
    steps_per_epoch = int(len(train_x) / TRAINING_BATCH_SIZE)
    H = model.fit_generator(
        data_gen_augmentation.flow(train_x, train_y, batch_size=TRAINING_BATCH_SIZE),
        validation_data=(test_x, test_y), steps_per_epoch=5,
        epochs=EPOCHS, verbose=1, callbacks=callbacks)

    # # save the model to disk
    print("[INFO] serializing network...")
    model.save('model.h5')  # creates a HDFfile 'model.h5'

    # Generate and save plots
    save_loss_plot(H, 'plot_loss.png')
    save_accuracy_plot(H, 'plot_accuracy.png')


train()
