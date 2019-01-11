from picamera import PiCamera
class Camera:
    def __init__(self, width, height):
        framerate = 90
        resolution = (width, height)
        camera = PiCamera()
        camera.resolution = resolution
        camera.framerate = framerate

        self.camera = camera

    def capture(self, output):
        self.camera.capture(output, format='rgb')

    def close(self):
        self.camera.close()