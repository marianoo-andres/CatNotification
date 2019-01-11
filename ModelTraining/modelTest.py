# import the necessary packages
import os
import time
from itertools import combinations

from math import factorial

from Model import Model

base_path = os.path.dirname(os.path.realpath(__file__))
model_base_path = os.path.join('Models', 'Best')
MODEL_NAMES = [os.path.join(model_base_path, 'best.h5'),
               os.path.join(model_base_path, 'bestV2.h5')
               ]


def testFolder(model, folderPath):
    test_y = []
    predictions = []
    # grab the image paths
    imageNames = os.listdir(folderPath)

    # loop over the input images and predict
    for imageName in imageNames:
        if not ".jpg" in imageName:
            continue
        # extract the class label from the image path
        type = imageName.split('_')[0]
        if type == 'gato':
            label = 1
        elif type == 'nogato':
            label = 0
        # True class
        test_y.append(label)
        # Prediction
        pred = model.predict(imagePath=os.path.join(folderPath, imageName))
        predictions.append(pred)

    image_count = {"gato": 0, "nogato_still": 0, "nogato_movement": 0}
    correct_pred_count = {}
    i = 0
    for imageName in imageNames:
        if ".jpg" not in imageName:
            continue
        if imageName.split("_")[0] == "nogato":
            if imageName.split("_")[1] == "still":
                key = "nogato_still"
            elif imageName.split("_")[1] == "movement":
                key = "nogato_movement"
        elif imageName.split("_")[0] == "gato":
            key = "gato"
        if key not in correct_pred_count:
            correct_pred_count[key] = 0
        image_count[key] += 1

        if test_y[i] == predictions[i][0]:
            correct_pred_count[key] += 1
        i += 1
    # Print stats
    total_correct = 0
    for key in correct_pred_count:
        total_correct += correct_pred_count[key]
        print("{} TOTAL: {} CORRECT: {} {}%".format(key, image_count[key], correct_pred_count[key],
                                                    100 * correct_pred_count[key] /
                                                    image_count[key]))
    return total_correct


def calculateNumberOfFiles(folderPath):
    count = 0
    for path, directories, files in os.walk(folderPath):
        for file in files:
            if "jpg" not in file:
                continue
            count += 1
    return count


def testSecuenciasFolders(models):
    model = Model(models)

    foldersPath = os.path.join("Data", "secuencias")
    folders = [
        "Gato",
        "NoGatoMovement",
        "NoGatoStill"
    ]
    totalCorrect = 0
    for folder in folders:
        folderPath = os.path.join(foldersPath, folder)
        folderCorrect = 0
        for secuencia in sorted(os.listdir(folderPath)):
            print("{}\t{}".format(folder, secuencia))
            secuenciaPath = os.path.join(folderPath, secuencia)
            correct = testFolder(model, secuenciaPath)
            totalCorrect += correct
            folderCorrect += correct
        totalFolderFiles = calculateNumberOfFiles(folderPath)
        print("\nTOTAL {} FILES: {}\tTOTAL CORRECT: {}\t ACCURACY: {}".format(folder,
                                                                              totalFolderFiles,
                                                                              folderCorrect,
                                                                              folderCorrect / totalFolderFiles))
    totalFiles = calculateNumberOfFiles(foldersPath)
    print("\n\nTOTAL FILES: {}\tTOTAL CORRECT: {}\t ACCURACY: {}".format(totalFiles, totalCorrect,
                                                                         totalCorrect / totalFiles))


# TEST ALL MODEL COMBINATIONS
#c = combinations(MODEL_NAMES, 2)
#n = factorial(len(MODEL_NAMES)) / (factorial(len(MODEL_NAMES) - 2) * 2)
#i = 0
#stats = {}
#for models in c:
#    i += 1
#    print(models)
#    model = Model(models)
#    print("TESTING {} of {}".format(i, n))
#    testFolder(model, os.path.join("Data", "Todo"))
#
# a = stats
# b = []
# for x in a.items():
#     b.append(x)
# b = sorted(b, key=lambda x: x[1], reverse=True)
# for x in b:
#     print(x)
# for modelName in os.listdir(model_base_path):
#     print(modelName)
#     test([os.path.join(model_base_path, modelName)])
#testSecuenciasFolders([os.path.join(model_base_path, 'best.h5')])

print("best")
models = [os.path.join(model_base_path, 'best.h5')]
model = Model(models)
testFolder(model, os.path.join("Data", "Todo"))
print("V2")
models = [os.path.join(model_base_path, 'bestV2.h5')]
model = Model(models)
testFolder(model, os.path.join("Data", "Todo"))
print("V2+best")
models = [os.path.join(model_base_path, 'bestV2.h5'),os.path.join(model_base_path, 'best.h5')]
model = Model(models)
testFolder(model, os.path.join("Data", "Todo"))




