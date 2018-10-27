import cv2
import glob
from matplotlib import pyplot as plt
faceDet = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
faceDet_two = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
faceDet_three = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
faceDet_four = cv2.CascadeClassifier("haarcascade_frontalface_alt_tree.xml")

fishface = cv2.face.FisherFaceRecognizer_create()
fishface.read('fish.xml')
emotions = ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"]


for files in glob.glob("C:\\Users\\HP\\Desktop\\classify\\*"):
    gray = cv2.imread(files)
    gray  = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    
    face = faceDet.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
    face_two = faceDet_two.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
    face_three = faceDet_three.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
    face_four = faceDet_four.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
    if len(face) == 1:
        facefeatures = face
    elif len(face_two) == 1:
        facefeatures = face_two
    elif len(face_three) == 1:
        facefeatures = face_three
    elif len(face_four) == 1:
        facefeatures = face_four
    else:
        facefeatures = ""
    for (x, y, w, h) in facefeatures:
        gray = gray[y:y+h, x:x+w]
        try:
            gray = cv2.resize(gray, (350, 350))
        except:
           pass
    
    plt.subplot(132)
    plt.title('img')
    plt.imshow(gray, 'gray')
    plt.xticks([])
    plt.yticks([])
    plt.show()
    Class, abc = fishface.predict(gray)
    print(emotions[Class])
