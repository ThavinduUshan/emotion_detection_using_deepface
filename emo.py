from deepface import DeepFace
import cv2
import matplotlib.pyplot as plt
import glob

folder_dir = "images"

open('log.txt', 'w').close() #clearing the content in the log file
logFile = open('log.txt', 'a')

for image in glob.iglob(f'{folder_dir}/*'):
   
    # check if the image ends with png or jpg
    if (image.endswith(".png") or image.endswith(".jpg")):
        img=cv2.imread(image)
        result = DeepFace.analyze(img, actions=['emotion'])
        logFile.write(f'{str(result)} \n')

logFile.close()
#img=cv2.imread('img2.jfif')
#result = DeepFace.analyze(img, actions=['age', 'gender', 'emotion'])
#sprint(result)
