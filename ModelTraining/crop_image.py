from PIL import Image
import os

def crop_image_from_center(source_path, new_width, new_height):
    image = Image.open(source_path)
    width, height = image.size

    left = (width - new_width) / 2
    top = (height - new_height) / 2
    right = (width + new_width) / 2
    bottom = (height + new_height) / 2
    image.crop((left, top, right, bottom)).save(source_path)

width = 224
height = 224
for path, directories, files in os.walk('RawImages'):
    for file in files:
        print(file)
        if "jpg" not in file:
            continue
        file_path = os.path.join(path,file)
        crop_image_from_center(file_path, width, height)