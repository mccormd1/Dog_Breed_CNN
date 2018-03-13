from keras.applications.resnet50 import ResNet50, preprocess_input as res_preprocess_input
from keras.preprocessing import image                  
import cv2
from keras.applications.inception_v3 import InceptionV3, preprocess_input as incep_preprocess_input
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense
from keras.models import Sequential
import numpy as np
from glob import glob      
import matplotlib.pyplot as plt    
import matplotlib.image as mpimg                    

face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')
dog_names = [item[20:-1] for item in sorted(glob("dogImages/train/*/"))]


def face_detector(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0

# define ResNet50 model
ResNet50_model = ResNet50(weights='imagenet')

def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def ResNet50_predict_labels(img_path):
    # returns prediction vector for image located at img_path
    img = res_preprocess_input(path_to_tensor(img_path))
    return np.argmax(ResNet50_model.predict(img))

def dog_detector(img_path):
    prediction = ResNet50_predict_labels(img_path)
    return ((prediction <= 268) & (prediction >= 151)) 
    


bottleneck_features = np.load('bottleneck_features/DogInceptionV3Data.npz')
train_inception = bottleneck_features['train']

inception_model = Sequential()
inception_model.add(GlobalAveragePooling2D(input_shape=train_inception.shape[1:]))
inception_model.add(Dense(64, activation='relu'))
inception_model.add(Dense(133, activation='softmax'))
inception_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
inception_model.load_weights('saved_models/weights.best.inceptionV3.hdf5')

# extract bottleneck features
def incept_predict_breed(img_path):

#     from keras.applications.inception_v3 import InceptionV3, preprocess_input
    bottleneck_feature = InceptionV3(weights='imagenet', include_top=False).predict(incep_preprocess_input(path_to_tensor(img_path)))

    predicted_vector = inception_model.predict(bottleneck_feature)
    # return dog breed that is predicted by the model
    return dog_names[np.argmax(predicted_vector)]
    

def dog_or_human(img_path):

    if face_detector(img_path):
        print('You look like a human')
    elif dog_detector(img_path):
        print('You look like a dog!')
        breed=incept_predict_breed(img_path)
        print('Specifically, you look like a',breed.replace('_',' '))
    else:
        print('I don\'t know what you are...')
    plt.imshow(mpimg.imread(img_path))


