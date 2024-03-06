#导入所需的包
import os
from PIL import Image
from tensorflow.keras import optimizers
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras import layers
from keras import models
#from keras import optimizers
plt.rcParams['font.sans-serif'] = ['SimHei']

#图片数据集可通过百度爬虫爬取（简单粗暴）
# 导入图片数据
# 一号虫图片
filedir = 'data/test1'
file_list1 = []
for root, dirs, files in os.walk(filedir):
    for file in files:
        if os.path.splitext(file)[1] == ".jpg":
            file_list1.append(os.path.join(root + '/', file))
# 批量改变图片像素
for filename in file_list1:
    try:
        im = Image.open(filename)
        im = im.convert('RGB')
        new_im = im.resize((128, 128))
        save_name = 'data/data1/' + filename[11:]
        new_im.save(save_name)
    except OSError as e:
        print(e.args)
# 重新建立新图像列表
filedir = 'data/data1'
file_list_1 = []
for root, dirs, files in os.walk(filedir):
    for file in files:
        if os.path.splitext(file)[1] == ".jpg":
            file_list_1.append(os.path.join(root + '/', file))
# 二号虫图片
filedir = 'data/test2'
file_list2 = []
for root, dirs, files in os.walk(filedir):
    for file in files:
        if os.path.splitext(file)[1] == ".jpg":
            file_list2.append(os.path.join(root + '/', file))

# 批量改变图片像素
for filename in file_list2:
    try:
        im = Image.open(filename)
        im = im.convert('RGB')
        new_im = im.resize((128, 128))
        save_name = 'data/data2/' + filename[11:]
        new_im.save(save_name)
    except OSError as e:
        print(e.args)
# 重新建立新图像列表
filedir = 'data/data2'
os.listdir(filedir)
file_list_2 = []
for root, dirs, files in os.walk(filedir):
    for file in files:
        if os.path.splitext(file)[1] == ".jpg":
            file_list_2.append(os.path.join(root + '/', file))

# 合并列表数据
file_list_all = file_list_1 + file_list_2
# 将图片转换为数组
M = []
for filename in file_list_all:
    im = Image.open(filename)
    width, height = im.size
    im_L = im.convert("L")
    Core = im_L.getdata()
    arr1 = np.array(Core, dtype='float32') / 255.0
    # arr1.shape
    list_img = arr1.tolist()
    M.extend(list_img)
x = np.array(M).reshape(len(file_list_all), width, height)
print(x.shape)
#设置图像标签
class_name = ['一号虫', '二号虫']
# 用字典存储图像信息
dict_label = {0: '一号虫', 1: '二号虫'}
print(dict_label[0])
print(dict_label[1])
# 用列表输入标签，0表示一号虫，1表示二号虫
label = [0] * len(file_list_1) + [1] * len(file_list_2)
y = np.array(label)
# 按照4：1的比例将数据集划分为训练集和测试集
train_images, test_imgages, train_labels, test_labels = train_test_split(x, y, test_size=0.2, random_state=0)
# 显示一张图片
# plt.figure()
# plt.imshow(train_images[1])
# plt.show()

#显示前30张图片
'''
plt.figure(figsize=(10,10))
for i in range(30):
    plt.subplot(5,6,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i])
    plt.xlabel(class_name[train_labels[i]])
plt.show()
'''

#构造神经网络并训练模型
model = models.Sequential()
model.add(layers.Flatten(input_shape=(128,128)))
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(2,activation='softmax'))

model.compile(optimizer = optimizers.RMSprop(lr=1e-4),
              loss = 'sparse_categorical_crossentropy',
              metrics = ['acc'])

history = model.fit(train_images,train_labels, epochs = 100)
history1 = model.fit(test_imgages,test_labels, epochs = 100)
test_loss,test_acc = model.evaluate(test_imgages,test_labels)
print('Test acc:',test_acc)
print(model.summary())

# #绘制训练精度和训练损失
# acc = history.history['acc']
# loss = history.history['loss']
# epochs = range(1,len(acc)+1)
# plt.plot(epochs, acc, 'bo', label='Training acc')
# plt.title('Training  accuracy')
# plt.legend()
# plt.figure()
# plt.plot(epochs, loss, 'bo', label='Training loss')
# plt.title('Training  loss')
# plt.legend()
# plt.show()
# #绘制测试精度和测试损失
# acc = history1.history['acc']
# loss = history1.history['loss']
# epochs = range(1,len(acc)+1)
# plt.plot(epochs, acc, 'bo', label='Test acc')
# plt.title('Test  accuracy')
# plt.legend()
# plt.figure()
# plt.plot(epochs, loss, 'bo', label='Test loss')
# plt.title('Test  loss')
# plt.legend()
# plt.show()

#预测一个图像
pre = model.predict(test_imgages)
print(dict_label[np.argmax(pre[2])])

#定义画图函数
def plot_image(i,pre_array,true_label,img):
    pre_array,true_label,img = pre_array[i],true_label[i],img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img)
    pre_label = np.argmax(pre_array)
    if pre_label == true_label:
        color = '#00bc57'
    else:
        color = 'red'   
    plt.xlabel("{} {:2.0f}% ({})".format(class_name[pre_label],
                                   100*np.max(pre_array),                                  
                                  class_name[true_label]),
                                   color= color )

def plot_value_array(i,pre_array,true_label):
    pre_array,true_label = pre_array[i],true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(len(class_name)),pre_array,
                        color = '#FF7F0E',width = 0.2 )   
    plt.ylim([0,1])
    pre_label = np.argmax(pre_array)
    thisplot[pre_label].set_color('red')
    thisplot[true_label].set_color('#00bc57')

#查看预测图像的真实标签和预测标签
i=1
plt.figure(figsize=(6,3))
#plt.subplot(1,2,1)
plot_image(i,pre,test_labels,test_imgages)
#plt.subplot(1,2,2)
#plot_value_array(i,pre,test_labels)
plt.show()