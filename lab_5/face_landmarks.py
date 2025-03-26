
import cv2
import dlib
import matplotlib.pyplot as plt

p = "shape_predictor_68_face_landmarks.dat"

def show_landmarks(path):
    img= cv2.imread(path)
    gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Inicjalizacja detektora
    detector = dlib.get_frontal_face_detector()
    # Detekcja
    faces = detector(gray)
    # Inicjalizacja predyktora keypointow
    predictor = dlib.shape_predictor(p)

    for face in faces:
        landmarks=predictor(img, face)
        for i in range(68):
            x = landmarks.part(i).x
            y = landmarks.part(i).y
            cv2.circle(img, (x,y), 2, (255,0,0), 1)
                
        x1=face.left()
        y1=face.top()
        x2=face.right()
        y2=face.bottom()
        cv2.rectangle(img, (x1,y1), (x2,y2),(0,0,255),5)

    cv2.imshow("Face lanndmarks",img)
    k = cv2.waitKey(0)
    if k==27:  # escap
        cv2.destroyAllWindows()

    #plt.imshow(img)

show_landmarks('face.jpg')