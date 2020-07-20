import os
from PIL import Image
from torch.utils import data
import numpy as np
from torchvision import transforms as T
import cv2
import random

class TCTDataset(data.Dataset):

    def __init__(self, root, balance=False,transforms=None, train=True, test=False):

        self.balance=balance
        self.test = test
        self.imgs = [os.path.join(root, img) for img in os.listdir(root)]
        if self.balance:
            normal_img=[]
            abnormal_img=[]
            for path in self.imgs:
                if path.split("/")[-1].split("_")[1]=="normal":
                    normal_img.append(path)
                else:
                    abnormal_img.append(path)
            num_normal=len(normal_img)
            abnormal_img=random.sample(abnormal_img,num_normal)
            self.imgs=normal_img+abnormal_img

        #TODO:normalize值中的mean和std的确定
        if transforms is None:#如果没有自己手动设置的transform，就是用默认的transform
            #normalize = T.Normalize(mean=[0, 0, 0],
                                    #std=[0, 0, 0])
            # 测试集和验证集
            if self.test or not train:
                self.transforms = T.Compose([
                    T.Resize(224),
                    T.CenterCrop(224),
                    T.ToTensor(),
                    #normalize
                ])
                # 训练集
            else:
                self.transforms = T.Compose([
                    T.Resize(256),
                    T.RandomCrop(224),
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    #normalize
                ])
        else:
            self.transforms=transforms

    def __getitem__(self, index):
        img_path = self.imgs[index]
        #TODO:根据图像名来定义label
        if img_path.split("/")[-1].split("_")[1]=="normal":
            label=0
        else:
            label=1
        data = Image.open(img_path)#对于读取图片尽量写在getitem里面，可以利用并行加速。
        # data=np.asarray(data)
        # data = cv2.imread(img_path)
        # data=Image.fromarray(data)
        data = self.transforms(data)
        return data, label

    def __len__(self):
        return len(self.imgs)

# tctdata=TCTDataset(root="/home/wangzehan/TCT_data/data_augment/train",balance=True,train=False)
# from torch.utils.data import DataLoader
# tctloader=DataLoader(tctdata,batch_size=11,shuffle=False)#,num_workers=8)
#
# print(len(tctdata))

# #data,label=list(iter(tctdata))[3]
# data,label=next(iter(tctloader))
# print(data.shape)
# # print(label)
