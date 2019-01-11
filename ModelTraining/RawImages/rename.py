import os
import uuid
names = [
[os.path.join('negra', 'test'),"gato_"],
[os.path.join('negra', 'train'),"gato_"],
[os.path.join('notnegra', 'test', 'movement'),"nogato_movement_"],
[os.path.join('notnegra', 'train', 'movement'),"nogato_movement_"],
[os.path.join('notnegra', 'test', 'still'),"nogato_still_"],
[os.path.join('notnegra', 'train', 'still'),"nogato_still_"],
]
for name in names:
    for path, directories, files in os.walk(name[0]):
        for file in files:
            if "jpg" not in file:
                continue
            old_path = os.path.join(path,file)
            new_name = name[1]+str(uuid.uuid4())+'.jpg'
            new_path = os.path.join(path,new_name)
            os.rename(old_path, new_path)