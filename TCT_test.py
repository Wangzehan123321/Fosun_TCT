import torch
import numpy as np
from TCT_dataset import TCTDataset
from torch.utils.data import DataLoader
import time
import math
import pretrainedmodels
import os
os.environ["CUDA_VISIBLE_DEVICES"]=str(2)


#定义模型
model_name="resnet18"
model=pretrainedmodels.__dict__[model_name](num_classes=1000,pretrained="imagenet")
#修改模型最后的输出类别
linear_TCT=torch.nn.Linear(in_features=512,out_features=2,bias=True)
model.last_linear=linear_TCT
model.load_state_dict(torch.load('../TCT_trainedmodels/resnet18/cslstm_99.tar'))#加载训练模型

#定义测试集
from TCT_dataset import TCTDataset
batch_size=64
num_workers=8
tsSet = TCTDataset("/home/wangzehan/TCT_data/data_process/train", train=False)
testDataloader = DataLoader(tsSet, batch_size,shuffle=True,num_workers=num_workers)

#评价指标
acc_all=0
batch_all=0
for ii, (image, label) in enumerate(testDataloader):
    score = model(image)
    acc_all+=torch.sum(torch.argmax(score,dim=1)==label).item()
    batch_all+=score.shape[0]
print(acc_all)
print(batch_all)
print("acc:{}".format(acc_all/batch_all))

