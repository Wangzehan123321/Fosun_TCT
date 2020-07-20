# from TCT_dataset import TCTDataset
# trSet = TCTDataset("/home/wangzehan/TCT_data/data_process/test", train=True)
# all=0
# num=0
# for data,label in trSet:
#     all+=1
#     if label==1:
#         num+=1
# print(all)
# print(num)
import torch
import numpy as np
from TCT_dataset import TCTDataset
from torch.utils.data import DataLoader
import time
import math
import pretrainedmodels
import os
os.environ["CUDA_VISIBLE_DEVICES"]=str(1)

#定义模型
model_name="resnet18"
model=pretrainedmodels.__dict__[model_name](num_classes=1000,pretrained="imagenet")
#修改模型最后的输出类别
linear_TCT=torch.nn.Linear(in_features=512,out_features=2,bias=True)
model.last_linear=linear_TCT
model.load_state_dict(torch.load('./cslstm_99.tar'))#加载训练模型

#注册forword_hook
input_feature=[]
def hook_fn_forward(module,input,output):
    input_feature.append(input[0])
    print(input[0].shape)

model.avgpool.register_forward_hook(hook_fn_forward)

#定义测试集
from TCT_dataset import TCTDataset
batch_size=1
num_workers=8
tsSet = TCTDataset("./TCT_data/data_process/val", train=False)
testDataloader = DataLoader(tsSet, batch_size,shuffle=True,num_workers=num_workers)

#测试
image, label=next(iter(testDataloader))
score = model(image)

last_linear_weight=list(model.parameters())[-2]
class_1=last_linear_weight[1:2,:]
print(input_feature[0].shape)
input_feature=input_feature[0].squeeze().view(512,49)
cam_feature=torch.mm(class_1,input_feature)
cam_feature=cam_feature.squeeze().view(7,7)
cam_feature=(cam_feature - cam_feature.min()) / (cam_feature.max() - cam_feature.min())
print(cam_feature)
import torchvision.transforms as T
from PIL import Image
cam_feature=T.ToPILImage()(cam_feature)
cam_feature=T.Resize((224,224))(cam_feature)
print(cam_feature.size)
print(cam_feature.getpixel((0,0)))
cam_feature.save("./img.jpg")
import cv2
img = cv2.imread('./img.jpg')
heatmap = cv2.applyColorMap(img, cv2.COLORMAP_JET)
image=T.ToPILImage()(image.squeeze())
import numpy
image=cv2.cvtColor(numpy.asarray(image),cv2.COLOR_RGB2BGR)
result = heatmap * 0.5 + image * 0.5
cv2.imwrite("./img_new.jpg", result)
# cv2.imwrite("./img_new.jpg",heatmap)
print(result.shape)
