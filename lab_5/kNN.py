import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import cv2
import dlib
import matplotlib.pyplot as plt
import os
import numpy as np

smiling = os.listdir('./archive/smile')
not_smiling = os.listdir('./archive/non_smile')

p = "shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(p)


def get_relative_landmarks(path):
    try:
        img= cv2.imread(path)
        gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        detector = dlib.get_frontal_face_detector()
        faces = detector(gray)
        lands=[]
    
        x1=faces[0].left()
        y1=faces[0].top()
        x2=faces[0].right()
        y2=faces[0].bottom()
        face_x_center = (x1+x2)/2
        face_y_center = (y1+y2)/2

        landmarks=predictor(img, faces[0])
        for i in range(68):
            x = landmarks.part(i).x
            y = landmarks.part(i).y
            x -= face_x_center
            y -= face_y_center

            lands.append((x,y))

        lands = np.array(lands).reshape(-1)
    except:
        lands = None
    return lands

smiling_data=[]

for path in smiling:
    a = get_relative_landmarks(os.path.join('./archive/smile', path))
    if a is not None:
        smiling_data.append(a)

non_smiling_data=[]

for path in not_smiling:
    a = get_relative_landmarks(os.path.join('./archive/non_smile', path))
    if a is not None:
        non_smiling_data.append(a)

smiling_data = np.vstack(smiling_data)
non_smiling_data = np.vstack(non_smiling_data)

print(smiling_data.shape, non_smiling_data.shape)

x = np.vstack((smiling_data, non_smiling_data))
y = np.concatenate((np.ones((smiling_data.shape[0])), np.zeros((non_smiling_data.shape[0]))))

print(x.shape, y.shape)

selector = SelectFromModel(estimator=GradientBoostingClassifier(), max_features=64).fit(x, y)

transformed_data = selector.transform(x)

print(transformed_data.shape)

transformed_data, x_test, y_train, y_test = train_test_split(transformed_data, y.reshape(-1,1))

model = KNeighborsClassifier(n_neighbors=3)
model.fit(transformed_data,y_train)

test_ = model.predict(x_test)
acc = accuracy_score(test_, y_test)

print(f'train_acc: {acc:.2f}')

test_sample = '/home/administrator3/Downloads/lab7(1)/archive/test/Alan_Ball_0002.jpg'

test_data = get_relative_landmarks(test_sample).reshape(1,-1)
test_data = selector.transform(test_data)

pred = model.predict(test_data)

print(f'prediction: {pred}')