import os
import uuid
names = [
	["Gato","gato_"],
	["NoGatoMovement", "nogato_movement_"],
	["NoGatoStill", "nogato_still_"]
]
for name in names:
    for path, directories, files in os.walk(name[0]):
        for file in files:
            if "jpg" not in file or "gato_" in file or "nogato_movement_" in file or "nogato_still_" in file:
                continue
            old_path = os.path.join(path,file)
            new_name = name[1]+file
            new_path = os.path.join(path,new_name)
            os.rename(old_path, new_path)

