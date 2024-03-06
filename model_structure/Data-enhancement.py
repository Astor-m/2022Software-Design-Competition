import PIL.Image as Image
import os
from torchvision import transforms as transforms
import string


root='data/'
def data():
    j=0
    dir=os.listdir(root)
    for name in dir:
        path=root+name
        imgList = os.listdir(path)
        for filename in imgList:
            im = Image.open(path+'/'+filename)
            im=im.convert("RGB")
            im.save(os.path.join(path,str(j)+'test.jpg'))

            #随机比例缩放
            new_im = transforms.Resize((100, 200))(im)
            print(f'{im.size}---->{new_im.size}')
            new_im.save(os.path.join(path, str(j)+'1.jpg'))

            #随机位置裁剪
            new_im = transforms.RandomCrop(100)(im)   # 裁剪出100x100的区域
            new_im.save(os.path.join(path, str(j)+'2_1.jpg'))
            new_im = transforms.CenterCrop(100)(im)
            new_im.save(os.path.join(path, str(j)+'2_2.jpg'))

            #水平/垂直翻转
            new_im = transforms.RandomHorizontalFlip(p=1)(im)   # p表示概率
            new_im.save(os.path.join(path, str(j)+'3_1.jpg'))
            new_im = transforms.RandomVerticalFlip(p=1)(im)
            new_im.save(os.path.join(path, str(j)+'3_2.jpg'))

            #随机角度翻转
            new_im = transforms.RandomRotation(45)(im)    #随机旋转45度
            new_im.save(os.path.join(path, str(j)+'4.jpg'))

            #色度、亮度、饱和度、对比度变化
            new_im = transforms.ColorJitter(brightness=1)(im)
            new_im = transforms.ColorJitter(contrast=1)(im)
            new_im = transforms.ColorJitter(saturation=0.5)(im)
            new_im = transforms.ColorJitter(hue=0.5)(im)
            new_im.save(os.path.join(path, str(j)+'5_1.jpg'))

            #随机灰度化
            new_im = transforms.RandomGrayscale(p=0.5)(im)    # 以0.5的概率进行灰度化
            new_im.save(os.path.join(path, str(j)+'6_2.jpg'))
            j=j+1

data()