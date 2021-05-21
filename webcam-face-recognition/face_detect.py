# Detect the face in video and capture and store it in Dataset folder 
# these faces will be used for training later for recognition by id
# each detect face will be given an id and will be stored and recognised with this id only 


# creating dataset
import cv2
import os

cam = cv2.VideoCapture(0)
cam.set(3, 640) # set video width
cam.set(4, 480) # set video height

# to detect the faces by openCV give path of haarcascade 
# According to the documentation of Cascade Classifier 
# It is a machine learning based approach where a cascade function is trained from a lot of 
# positive and negative images. It is then used to detect objects in other images.
face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# for each person, enter one numeric face id
# for recoginsing it later
face_id = input('\n Enter user id and press enter')
print("\n [INFO] Intializing face capture.")

# Initialize individual sampling face count
count = 0
while(True):
    ret,img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)
# faces are detected as pixels
    for(x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)
        count += 1
        cv2.imwrite("Dataset/User."+str(face_id)+'.'+str(count)+".jpg", gray[y:y+h,x:x+w])
        cv2.imshow('image', img)
    # how mush time user need to wait to Save the captured image into the datsets folder
    k=cv2.waitKey(100) & 0xff
    # to increase the efficiency
    # if waiting time reaches 27 secs breaks out of loop and store the capture img.
    if(k==27):
        break
    # or if the count of img reaches to 30
    elif count>=30:
        break

# Do a bit of cleanup
print("\n [INFO] Exiting program")
cam.release()
cv2.destroyWindows()
