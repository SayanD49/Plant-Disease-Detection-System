import tensorflow as tf
import numpy as np
import json
import cv2

model = tf.keras.models.load_model("model/model.h5")

with open("model/classes.json") as f:
    class_dict = json.load(f)

classes = list(class_dict.keys())

def predict_disease(img_path):

    img = cv2.imread(img_path)
    img = cv2.resize(img,(224,224))

    img = img/255.0

    img = np.expand_dims(img,axis=0)

    pred = model.predict(img)

    index = np.argmax(pred)

    confidence = float(np.max(pred))*100

    return classes[index],confidence