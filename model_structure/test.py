from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import load_img


import numpy as np
from keras.models import Sequential
from keras.layers import *
from tensorflow.keras import optimizers
from sklearn.metrics import recall_score
import keras
from keras.callbacks import EarlyStopping
early_stopping_monitor = EarlyStopping(patience=2)

x_train = np.random.random((100, 100, 100, 3))
# 100张图片，每张100*100*3
y_train = keras.utils.np_utils.to_categorical(np.random.randint(10, size=(100, 1)), num_classes=10)
# 100*10
x_test = np.random.random((20, 100, 100, 3))
y_test = keras.utils.np_utils.to_categorical(np.random.randint(10, size=(20, 1)), num_classes=10)



def vgg8():
    model = Sequential()
    # input: 100x100 images with 3 channels -> (100, 100, 3) tensors.
    # this applies 32 convolution filters of size 3x3 each.
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics= ['accuracy']) 
    model.fit(x_train,y_train, validation_split=0.33, epochs=10, callbacks= [early_stopping_monitor], verbose=False)

    

    return model

model=vgg8()
acc=model.history.history['accuracy']
val_acc=model.history.history['val_accuracy']
loss=model.history.history['loss']
val_loss=model.history.history['val_loss']
epochs=range(len(acc)) # Get number of epochs
plt.plot(epochs, acc, 'r')
plt.plot(epochs, val_acc, 'b')
plt.title('Training and validation accuracy')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend(["Accuracy", "Validation Accuracy"])

plt.figure()

# 画loss曲线
plt.plot(epochs, loss, 'r')
plt.plot(epochs, val_loss, 'b')
plt.title('Training and validation loss')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(["Loss", "Validation Loss"])


plt.show()