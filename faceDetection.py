#import libraries
import os   
import cv2
import numpy

#get the path of the directory
dir_path = os.path.dirname(os.path.realpath(__file__))
#create the Output folder if it does not exist
if not os.path.exists('Output'): 
  os.makedirs('Output')
#import the models provided in the OpenCV repository
model = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'weights.caffemodel')
#loop through all the files in the folder
for file in os.listdir(dir_path):
  #split the file name and the extension into two variales
  filename, file_extension = os.path.splitext(file)
  #check if the file extension is .png,.jpeg or .jpg to avoid reading other files in the directory
  if (file_extension in ['.png','.jpg','.jpeg']):
    #read the image using cv2
    image = cv2.imread(file)
    #accessing the image.shape tuple and taking the first two elements which are height and width
    (h, w) = image.shape[:2]
    #get our blob which is our input image after mean subtraction, normalizing, and channel swapping
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    #input the blob into the model and get back the detections from the page using model.forward()
    model.setInput(blob)
    detections = model.forward()
    #Iterate over all of the faces detected and extract their start and end points
    count = 0
    for i in range(0, detections.shape[2]):
      box = detections[0, 0, i, 3:7] * numpy.array([w, h, w, h])
      (startX, startY, endX, endY) = box.astype("int")

      confidence = detections[0, 0, i, 2]
      #if the algorithm is more than 16.5% confident that the detection is a face, show a rectangle around it
      if (confidence > 0.165):
        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
        count = count + 1
    #save the modified image to the Output folder
    cv2.imwrite('Output/' + file, image)
    #print out a success message
    print("Face detection complete for image "+ file + " ("+ str(count) +") faces found!")

