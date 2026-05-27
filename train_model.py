import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense,GlobalAveragePooling2D
from tensorflow.keras.models import Model
import json

IMG_SIZE = 224
BATCH = 32

train_dir = "dataset/train"
valid_dir = "dataset/valid"

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

valid_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE,IMG_SIZE),
    batch_size=BATCH,
    class_mode="categorical"
)

val_data = valid_datagen.flow_from_directory(
    valid_dir,
    target_size=(IMG_SIZE,IMG_SIZE),
    batch_size=BATCH,
    class_mode="categorical"
)

base_model = MobileNetV2(
    input_shape=(224,224,3),
    include_top=False,
    weights="imagenet"
)

for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)

predictions = Dense(train_data.num_classes,activation="softmax")(x)

model = Model(inputs=base_model.input,outputs=predictions)

model.compile(
optimizer="adam",
loss="categorical_crossentropy",
metrics=["accuracy"]
)

model.fit(train_data,validation_data=val_data,epochs=5)

model.save("model/model.h5")

with open("model/classes.json","w") as f:
    json.dump(train_data.class_indices,f)

print("Model trained successfully")