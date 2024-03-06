from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras import optimizers
import tensorflow as tf

def vgg8(x_train,y_train):
    model = Sequential()
    # input: 224x224 images with 3 channels -> (224, 224, 3) tensors.
    # this applies 32 convolution filters of size 3x3 each.
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    opt = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-06)
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer=opt,metrics=['accuracy'])

    model.fit(x_train, y_train, batch_size=32, epochs=20)
    #score = model.evaluate(x_test, y_test, batch_size=32)

    return model

    
def vgg(x_train,y_train):
    model = Sequential()
    # input: 224x224 images with 3 channels -> (224, 224, 3) tensors.
    # this applies 32 convolution filters of size 3x3 each.
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    opt = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-06)
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer=opt,metrics=['accuracy'])

    model.fit(x_train, y_train, batch_size=32, epochs=20)
    #score = model.evaluate(x_test, y_test, batch_size=32)

    return model

def vgg16(x_train,y_train):
    model = Sequential()
    model.add(Conv2D(64, (3, 3), strides=(1, 1), input_shape=(224, 224, 3), padding='same', activation='relu', kernel_initializer='uniform'))
    model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    model.add(Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='softmax'))
    sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer=sgd, metrics=['accuracy'])

    model.fit(x_train, y_train, batch_size=32, epochs=10)
    # model.summary()
    return model
 
 
def AlexNet(x_train,y_train):
    model = Sequential()
    # 第一层
    model.add(Conv2D(filters=96, kernel_size=(11, 11),
                     strides=(4, 4), padding='valid',
                     input_shape=(224, 224, 3),
                     activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3, 3),
                           strides=(2, 2),
                           padding='valid'))
    # 第二层
    model.add(Conv2D(filters=256, kernel_size=(5, 5),
                     strides=(1, 1), padding='same',
                     activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3, 3),
                           strides=(2, 2),
                           padding='valid'))
    # 第三层
    model.add(Conv2D(filters=384, kernel_size=(3, 3),
                     strides=(1, 1), padding='same',
                     activation='relu'))
    model.add(Conv2D(filters=384, kernel_size=(3, 3),
                     strides=(1, 1), padding='same',
                     activation='relu'))
    model.add(Conv2D(filters=256, kernel_size=(3, 3),
                     strides=(1, 1), padding='same',
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3),
                           strides=(2, 2), padding='valid'))
    # 第四层
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
 
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
 
    model.add(Dense(1000, activation='relu'))
    model.add(Dropout(0.5))
 
    # Output Layer
    model.add(Dense(10,activation='softmax'))
    # model.add(Activation('softmax'))
    sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer=sgd)

    model.fit(x_train, y_train, batch_size=32, epochs=20)

    return model

