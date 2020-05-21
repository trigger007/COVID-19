# -*- coding: utf-8 -*-
"""
Created on Mon May  4 21:36:46 2020

@author: Shreyansh Satvik
"""

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

image_path="dataset"

#setting hyperparameters like learning rate, no of epochs and batch_size
INIT_LR=1e-4
EPOCHS=20
BS=32
#grabbing all imagepath in dataset
image_dataset=list(paths.list_images(image_path))
data=[]
labels=[]

for imagepath in image_dataset:
    #extracting the class
    label= imagepath.split(os.path.sep)[-2]
    
    #loading image in 224x224 and preprocess it as model requires 224x224 input
    image= load_img(imagepath,target_size=(224,224))
    image=img_to_array(image)
    image = preprocess_input(image)
    
    #update labels and data respectively
    data.append(image)
    labels.append(label)
    
#convert the image to numpy array

data=np.array(data,dtype="float32")
labels=np.array(labels)

#NOTE THIS STRATEGY FOR DATASET CONVERSION IS ONLY FOR SMALL DATASET FOR LARGER DATASET USE HDF5 FOR BETTER USE

#ONE HOT ENCODING FOR LABELS

lb=LabelBinarizer()
labels=lb.fit_transform(labels)
labels=to_categorical(labels)

#performing train-test split
(trainX,testX,trainY,testY)=train_test_split(data,labels,test_size=0.20,stratify=labels,random_state=42)

#comstruct more training image using dataaugmentation
aug= ImageDataGenerator(rotation_range=20,zoom_range=0.15,width_shift_range=0.2,shear_range=0.15,horizontal_flip=True,fill_mode="nearest")

#loading mobilenetV2 model
baseModel = MobileNetV2(weights="imagenet", include_top=False,input_tensor=Input(shape=(224, 224, 3)))

baseModel=MobileNetV2(alpha=1.0, include_top=False, weights='imagenet', input_tensor=Input(shape=(224, 224, 3)))

headmodel=baseModel.output
headmodel=AveragePooling2D(pool_size=(7,7))(headmodel)
headmodel=Flatten(name="flatten")(headmodel)
headmodel=Dense(128,activation="relu")(headmodel)
headmodel=Dropout(0.5)(headmodel)
headmodel=Dense(2,activation="softmax")(headmodel)

model=Model(inputs=baseModel.input,outputs=headmodel)

#loop over all basemodel and freeze them

for layer in baseModel.layers:
    layer.trainable=False
    
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])

H = model.fit(
	aug.flow(trainX, trainY, batch_size=BS),
	steps_per_epoch=len(trainX) // BS,
	validation_data=(testX, testY),
	validation_steps=len(testX) // BS,
	epochs=EPOCHS)
#predicting on test set
predtest = model.predict(testX, batch_size=BS)

# for each image in the testing set we need to find the index of the
# label with corresponding largest predicted probability
predtest = np.argmax(predtest, axis=1)
# show a nicely formatted classification report
print(classification_report(testY.argmax(axis=1), predtest,
	target_names=lb.classes_))
# serialize the model to disk
print("[INFO] saving mask detector model...")
model.save("maskdetector.model", save_format="h5")

#plot
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])