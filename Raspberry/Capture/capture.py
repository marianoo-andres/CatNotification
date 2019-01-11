import datetime
import numpy as np
import time
import picamera
import io
from PIL import Image
import os

def crop_image_and_save(image, path, top, left, size):
    right = left + size
    bottom = top + size
    image.crop((left, top, right, bottom)).save(path)

camera = picamera.PiCamera()
camera.resolution = (350, 350)
camera.framerate = 90
count = 1
time.sleep(2)
while True:
    camera.capture('temp.jpg', resize=(350,350))
    image = Image.open('temp.jpg')
    t = datetime.datetime.now()-datetime.timedelta(hours=3)
    t = t.strftime('%Y-%m-%d_%H-%M-%S.%f')
    filepath = os.path.join('images', '{}_{:03d}.jpg'.format(t,count))
    crop_image_and_save(image, filepath, 110, 87, 224)
    print('Captured %s' % filepath)
    count += 1
    time.sleep(1)
camera.stop()
