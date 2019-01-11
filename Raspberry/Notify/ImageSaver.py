import os

class ImageSaver:
    def __init__(self):
        self.imageNumberGato = self.getImageNumber("Gato")
        self.imageNumberNoGato = self.getImageNumber("NoGato")

    def getImageNumber(self, folder):
        max = 0
        for filename in os.listdir(os.path.join("Images", folder)):
            imageNumber = int(filename.split("_")[2])
            if imageNumber > max:
                max = imageNumber
        return max + 1

    def getImageSavePath(self, prediction):
        GATO = 1
        NOGATO = 0
        label, probability = prediction
        t = datetime.datetime.now() - datetime.timedelta(hours=3)
        t = t.strftime('%Y-%m-%d_%H-%M-%S.%f')
        if label == GATO:
            folder = "Gato"
            imageName = "{}_{}_{}.jpg".format(t, self.imageNumberGato, probability)
            self.imageNumberGato += 1
        elif label == NOGATO:
            folder = "NoGato"
            imageName = "{}_{}_{}.jpg".format(t, self.imageNumberNoGato, probability)
            self.imageNumberNoGato += 1
        imagePath = os.path.join("Images", folder, imageName)
        return imagePath

    def save(self, image, prediction):
        imagePath = self.getImageSavePath(prediction)
        image.save(imagePath)