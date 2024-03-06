import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.layers import *
from keras.layers.advanced_activations import PReLU
from tensorflow.keras import optimizers
import matplotlib.pyplot as plt





#多分类问题VGG
# Generate dummy data

x_train = np.random.random((100, 100, 100, 3))
# 100张图片，每张100*100*3
y_train = keras.utils.np_utils.to_categorical(np.random.randint(10, size=(100, 1)), num_classes=10)
# 100*10
x_test = np.random.random((20, 100, 100, 3))
y_test = keras.utils.np_utils.to_categorical(np.random.randint(10, size=(20, 1)), num_classes=10)
# 20*100
# plt.figure()
# plt.imshow(x_train[0])
# plt.show()

def vgg8(x_train,y_train):
    model = Sequential()
    # input: 100x100 images with 3 channels -> (100, 100, 3) tensors.
    # this applies 32 convolution filters of size 3x3 each.
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)))
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

    sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd)

    model.fit(x_train, y_train, batch_size=32, epochs=10)
    #score = model.evaluate(x_test, y_test, batch_size=32)

    return model





# #获取单张图片
# def get_image(rootdir):    
# 	fs = []    
# 	for root, dirs, files in os.walk(rootdir,topdown = True):
# 		for name in files: 
# 			_, ending = os.path.splitext(name)
# 			if ending == ".jpg":
# 				fs.append(os.path.join(root,name))   
# 	return fs


# #获取
# def preprocess(filepath):
#     pass




# def keras_batchnormalization_relu(layer):
#     BN = BatchNormalization()(layer)
#     ac = PReLU()(BN)
#     return ac
 
 
# def AlexNet(resize=227, classes=10):
#     model = Sequential()
#     # 第一层
#     model.add(Conv2D(filters=96, kernel_size=(11, 11),
#                      strides=(4, 4), padding='valid',
#                      input_shape=(resize, resize, 3),
#                      activation='relu'))
#     model.add(BatchNormalization())
#     model.add(MaxPooling2D(pool_size=(3, 3),
#                            strides=(2, 2),
#                            padding='valid'))
#     # 第二层
#     model.add(Conv2D(filters=256, kernel_size=(5, 5),
#                      strides=(1, 1), padding='same',
#                      activation='relu'))
#     model.add(BatchNormalization())
#     model.add(MaxPooling2D(pool_size=(3, 3),
#                            strides=(2, 2),
#                            padding='valid'))
#     # 第三层
#     model.add(Conv2D(filters=384, kernel_size=(3, 3),
#                      strides=(1, 1), padding='same',
#                      activation='relu'))
#     model.add(Conv2D(filters=384, kernel_size=(3, 3),
#                      strides=(1, 1), padding='same',
#                      activation='relu'))
#     model.add(Conv2D(filters=256, kernel_size=(3, 3),
#                      strides=(1, 1), padding='same',
#                      activation='relu'))
#     model.add(MaxPooling2D(pool_size=(3, 3),
#                            strides=(2, 2), padding='valid'))
#     # 第四层
#     model.add(Flatten())
#     model.add(Dense(4096, activation='relu'))
#     model.add(Dropout(0.5))
 
#     model.add(Dense(4096, activation='relu'))
#     model.add(Dropout(0.5))
 
#     model.add(Dense(1000, activation='relu'))
#     model.add(Dropout(0.5))
 
#     # Output Layer
#     model.add(Dense(classes,activation='softmax'))
#     # model.add(Activation('softmax'))
#     sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
#     model.compile(loss='categorical_crossentropy', optimizer=sgd)

#     return model




def vgg16(resize=224, classes=10, prob=0.5):
    model = Sequential()
    model.add(Conv2D(64, (3, 3), strides=(1, 1), input_shape=(resize, resize, 3), padding='same', activation='relu',
                     kernel_initializer='uniform'))
    model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 2), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
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
    model.add(Dropout(prob))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(prob))
    model.add(Dense(classes, activation='softmax'))
    sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd)

    return model




