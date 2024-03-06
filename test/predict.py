from tkinter import Image
import numpy as np
from keras.models import load_model
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
 
model = load_model('model/my_model.h5')
#model.summary()#输出模型

#改变测试图片分辨率
testfile='model/data/data.jpg'
im = Image.open(testfile)
im = im.convert('RGB')
new_im = im.resize((255, 255))
save_name = 'model/data/test.jpg'
new_im.save(save_name)

#将测试图片转为数组
im = Image.open(save_name)
width, height = im.size
im_L = im.convert("L")
Core = im_L.getdata()
arr1 = np.array(Core, dtype='float32') / 255.0
# arr1.shape
list_img = arr1.tolist()
#z = list_img.reshape(len(file_list_all), width, height)

#model.predict()参数只能是
Z=[]
Z.append(list_img)


dict_label = {0: '草履蚧', 1: '麻皮蝽'}
#预测图片
pre = model.predict(Z)
print(dict_label[np.argmax(pre)])#查看预测类型




   
