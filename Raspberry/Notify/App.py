import time

print("Loading modules...")
start = time.time()
import numpy as np
from Model import Model
from PIL import Image
import os
import datetime
from Camera import Camera
from Notificator import Notificator
from ImageSaver import ImageSaver
print("Modules loaded in {} seconds".format(time.time() - start))


class App:
    def cropImage(self, image, top, left, size):
        right = left + size
        bottom = top + size
        return image.crop((left, top, right, bottom))

    def start(self):
        # Do not change
        width = 352
        height = 352
        # output to capture image
        output = np.empty((height, width, 3), dtype=np.uint8)
        # Init camera
        camera = Camera(width, height)
        # Init image saver
        imageSaver = ImageSaver()
        # Init model
        model = self.getModel()
        # Init notificator
        notificator = Notificator()
        # Start predicting
        tempCount = 1
        while True:
            #print("Step...")
            #start = time.time()
            # Capture image to output
            camera.capture(output)
            imageToCrop = Image.fromarray(output)
            image = self.cropImage(imageToCrop, 110, 87, 224)
            tempName = 'temp{}.jpg'.format(tempCount)
            image.save(tempName)
            tempCount += 1
            if tempCount > Notificator.PREDICTIONS_LENGTH:
                tempCount = 1
            # Predict
            prediction = model.predict(imagePath=tempName)
            # Save image to disk with label and probability
            imageSaver.save(image, prediction)
            # Manage Notification
            notificator.manageNotification(prediction)
            #print("Step took {} seconds".format(time.time()-start))

        camera.close()

    def getModel(self):
        print("Loading model...")
        start = time.time()
        basePath = os.path.join('Models')
        MODEL_NAMES = [os.path.join(basePath, 'best.h5')]
        model = Model(MODEL_NAMES)
        print("Model loaded in {} seconds".format(time.time() - start))
        return model


# Launch code
app = App()
app.start()
