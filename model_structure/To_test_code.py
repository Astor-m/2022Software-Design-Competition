from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import load_img


import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten,Activation,BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import *
# from keras.layers.advanced_activations import PReLU
from tensorflow.keras import optimizers
from sklearn.metrics import recall_score
import ntpath
from keras.callbacks import EarlyStopping
early_stopping_monitor = EarlyStopping(patience=2)


dir='dataset/'
traindir='trainset/'
filelist=[]#存储处理后图像地址
class_dir=[]
class_num=[]
label=[]#标签


#获取类别文件夹
class_name=os.listdir(dir)
class_dir=class_dir+class_name


#获取图片地址
def get_image(classdir):
    file_list=[]
    path=dir+classdir
    imgList = os.listdir(path)
    for x in imgList:
        file_list.append(path+"/"+x)
    return file_list




# 批量改变图片像素
def unify_pixel():
    for classdir in class_dir:#读取图片地址
        num=0
        file_list=get_image(classdir)
        for filename in file_list:
            
            num=num+1
            try:
                new_im = load_img(filename, target_size=(32, 32))
                save_name = ntpath.basename(filename)
                new_im.save(traindir+classdir+"/"+save_name)
                filelist.append(traindir+classdir+"/"+save_name)
            except OSError as e:
                print(e.args)
            
        class_num.append(num)
    return filelist

filelist=unify_pixel()#获取数据集图片地址

#将训练集图片转换为数组
def img_to_num():
    M = []
    for filename in filelist:
        im = Image.open(filename)
        width, height = im.size
        im_RGB = im.convert("RGB")
        Core = im_RGB.getdata()
        arr1 = np.array(Core, dtype='float32') / 255.0
        # arr1.shape
        list_img = arr1.tolist()
        M.extend(list_img)
    x = np.array(M).reshape(len(filelist), width, height,3)
    return x

x = img_to_num()#获取

i=0
for number in class_num:
    label = label+([i] * number)
    i=i+1
y = np.array(label)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=0)




def net():
    model = Sequential()#初始化模型
    
    #第一层
    model.add(Conv2D(64, (3, 3), strides=(1, 1), input_shape=(32, 32, 3), padding='same', activation='relu', kernel_initializer='uniform'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.5))

    #第二层
    model.add(Conv2D(128, (3, 3), strides=(1, 1)))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    

    #全连接层
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    
    #分类层
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(10, activation='softmax'))


    opt= optimizers.RMSprop(lr=0.003, rho=0.9, epsilon=1e-06)
    # model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer=opt,metrics=['accuracy'])

    model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics= ['accuracy']) 
    
    return model




Mymodel=net()
Mymodel.fit(x_train, y_train, batch_size=32, validation_split=0.33,epochs=200)
#Mymodel.fit(x_train,y_train, epochs=3, callbacks= [early_stopping_monitor], verbose=False)

#绘制曲线

acc=Mymodel.history.history['accuracy']
val_acc=Mymodel.history.history['val_accuracy']
loss=Mymodel.history.history['loss']
val_loss=Mymodel.history.history['val_loss']

print(acc)
print(val_acc)
print(loss)
print(val_loss)



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


score = Mymodel.evaluate(x_test, y_test, batch_size=32)

Mymodel.save("model.h5")



