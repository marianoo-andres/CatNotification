import numpy as np
from keras.engine.saving import load_model
from keras.utils import CustomObjectScope
from keras_applications import mobilenet_v2
from keras_preprocessing.image import load_img, img_to_array


class Model:
    IMAGE_SIZE = 32

    def __init__(self, modelNames):
        self.models = []
        for name in modelNames:
            with CustomObjectScope({'relu6': mobilenet_v2.relu6}):
                model = load_model(name)
            self.models.append(model)
        # for model in self.models:
        #     model.summary()
        #     exit(1)

    def predict(self, arrayImage=None, imagePath=None):
        GATO = 1
        NOGATO = 0
        if imagePath:
            image = load_img(imagePath, target_size=(Model.IMAGE_SIZE, Model.IMAGE_SIZE))
            arrayImage = self.pilImageToArray(image)

        else:
            arrayImage = self.normalizeArray(arrayImage)
        probs = []
        for model in self.models:
            probs.append(model.predict(arrayImage)[0])
        lenModels = len(self.models)
        sum = 0
        for x in range(lenModels):
            sum += probs[x][0]
        nogatoProb = sum / lenModels
        sum = 0
        for x in range(lenModels):
            sum += probs[x][1]
        gatoProb = sum / lenModels

        if gatoProb > nogatoProb:
            return (GATO, gatoProb)
        return (NOGATO, nogatoProb)

    def pilImageToArray(self, pilImage):
        # Convert image to numpy array
        arrayImage = img_to_array(pilImage)
        # Normalize image
        arrayImage = arrayImage / 255.0
        # Expand dim as classifier expects and array of images
        arrayImage = np.expand_dims(arrayImage, axis=0)
        return arrayImage

    def normalizeArray(self, arrayImage):
        # Normalize image
        arrayImage = arrayImage / 255.0
        # Expand dim as classifier expects and array of images
        arrayImage = np.expand_dims(arrayImage, axis=0)
        return arrayImage