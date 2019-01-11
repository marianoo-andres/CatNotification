import os
count = 0
for path, directories, files in os.walk(os.path.join('asd')):
    for file in files:
        if ".jpg" not in file:
            continue
        count += 1
        if not count % 9 == 0:
            file_path = os.path.join(path,file)
            os.remove(file_path)