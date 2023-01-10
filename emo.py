from deepface import DeepFace
import cv2
import matplotlib.pyplot as plt
import glob

#folder in which the images are stored
folder_dir = "images"

#to capture the dominant emotion counts of the image set
emotion_count = {
  'happy' : 0,
  'sad' : 0,
  'neutral' : 0,
  'surprise' : 0,
  'disgust' : 0,
  'angry' : 0,
  'fear' : 0,
}

open('log.txt', 'w').close() #clearing the content in the log file
logFile = open('log.txt', 'a')

i = 1 #to capture the record number in the log file

for image in glob.iglob(f'{folder_dir}/*'):
   
    # check if the image ends with png or jpg
    if (image.endswith(".png") or image.endswith(".jpg")):
        img=cv2.imread(image)
        result = DeepFace.analyze(img, actions=['emotion']) #analyzing the emotion
        
        #get the dominant emotion from the result 
        dEmo = result['dominant_emotion']
        
        #increasing the emotion count based on the dominant emotion
        if(dEmo == 'happy') :
          emotion_count['happy'] += 1
        elif(dEmo == 'sad') : 
          emotion_count['sad'] += 1
        elif(dEmo == 'neutral') : 
          emotion_count['neutral'] += 1
        elif(dEmo == 'surprise') : 
          emotion_count['surprise'] += 1
        elif(dEmo == 'disgust') : 
          emotion_count['disgust'] += 1
        elif(dEmo == 'angry') : 
          emotion_count['angry'] += 1
        elif(dEmo == 'fear') : 
          emotion_count['fear'] += 1
        
        logFile.write(f'{i}. {str(result)} \n') # writing the result to the log file.
        i += 1 #increment the record number

#writing the dominant emotion counts to the log file at the end
logFile.write(f'\n\n Dominant Emotion Counts : {str(emotion_count)} \n')
#closing the file
logFile.close()
