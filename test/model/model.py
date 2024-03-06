#导入所需的包
import os
import tensorflow as tf
import h5py
from PIL import Image
from tensorflow.keras import optimizers
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras import layers
from keras import models
plt.rcParams['font.sans-serif'] = ['SimHei']

#图片数据集可通过百度爬虫爬取（简单粗暴）
# 导入图片数据
# 一号虫图片，获取图片路径
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
        new_im = im.resize((255, 255))
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
        new_im = im.resize((255, 255))
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
class_name = ['草履蚧', '麻皮蝽']
# 用字典存储图像信息
dict_label = {0: '草履蚧', 1: '麻皮蝽'}
# 用列表输入标签，0表示一号虫，1表示二号虫
label = [0] * len(file_list_1) + [1] * len(file_list_2)
y = np.array(label)

print(y)
print("dasda")

#此处省略，直接将全部图片作为训练集
train_images, test_imgages, train_labels, test_labels = train_test_split(x, y, test_size=0.2, random_state=0)
#train_images：训练集
#test_imgages：测试集
#train_labels：训练集图片类别
#test_labels：测试集图片类别

#print(train_labels,test_labels)

# plt.figure()
# plt.imshow(x[1])
# plt.show()



#构造神经网络并训练模型model
model = models.Sequential()#初始化模型
model.add(layers.Flatten(input_shape=(255,255)))#设置图片输入大小
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(2,activation='softmax'))

model.compile(optimizer = optimizers.RMSprop(lr=1e-4),
              loss = 'sparse_categorical_crossentropy',
              metrics = ['acc'])


model.summary()



history = model.fit(x,y, epochs = 100)#训练模型
# history1 = model.fit(test_imgages,test_labels, epochs = 50)
# test_loss,test_acc = model.evaluate(test_imgages,test_labels)
# print('Test acc:',test_acc)
# print(model.summary())

model.save("my_model.h5")
#f=h5py.File('mymodel','w')
#测试预测图片
# #改变测试图片分辨率
# testfile='data/data.jpg'
# im = Image.open(testfile)
# im = im.convert('RGB')
# new_im = im.resize((255, 255))
# save_name = 'data/test.jpg'
# new_im.save(save_name)

# #将测试图片转为数组
# im = Image.open(save_name)
# width, height = im.size
# im_L = im.convert("L")
# Core = im_L.getdata()
# arr1 = np.array(Core, dtype='float32') / 255.0
# # arr1.shape
# list_img = arr1.tolist()
# #z = list_img.reshape(len(file_list_all), width, height)

# #model.predict()参数只能是
# Z=[]
# Z.append(list_img)


# #预测图片
# pre = model.predict(Z)
# print(dict_label[np.argmax(pre)])#查看预测类型



# #定义画图函数
# def plot_image(pre_array,true_label,img):
#     plt.grid(False)
#     plt.xticks([])
#     plt.yticks([])
#     plt.imshow(np.real(img))
#     #pre_label = np.argmax(pre_array)
#     if pre_array == true_label:
#         color = '#00bc57'
#     else:
#         color = 'red'
#     plt.xlabel("{} {:2.0f}% ({})".format(class_name[pre_array],
#                                    100*np.max(pre_array),
#                                   class_name[true_label]),
#                                    color= color )

# def plot_value_array(i,pre_array,true_label):
#     pre_array,true_label = pre_array[i],true_label[i]
#     plt.grid(False)
#     plt.xticks([])
#     plt.yticks([])
#     thisplot = plt.bar(range(len(class_name)),pre_array,
#                         color = '#FF7F0E',width = 0.2 )
#     plt.ylim([0,1])
#     pre_label = np.argmax(pre_array)
#     thisplot[pre_label].set_color('red')
#     thisplot[true_label].set_color('#00bc57')

#查看预测图像的真实标签和预测标签
#i=0
#plt.figure(figsize=(6,3))
#plt.subplot(1,2,1)
#Z.append(1)
#plot_image(pre[0],2,Z)
#plt.subplot(1,2,2)
#plot_value_array(0,pre,z)
#plt.show()

# def predict_image(filename):
#     im = Image.open(filename)
#     im = im.convert('RGB')
#     new_im = im.resize((255, 255))
#     save_name = 'data/predict.jpg'
#     new_im.save(save_name)

#     #将测试图片转为数组
#     im = Image.open(save_name)
#     width, height = im.size
#     im_L = im.convert("L")
#     Core = im_L.getdata()
#     arr1 = np.array(Core, dtype='float32') / 255.0
#     # arr1.shape
#     list_img = arr1.tolist()
#     #z = list_img.reshape(len(file_list_all), width, height)

#     #model.predict()参数只能是
#     Z=[]
#     Z.append(list_img)
#     pre = model.predict(Z)
#     print(dict_label[np.argmax(pre)])#查看预测类型


