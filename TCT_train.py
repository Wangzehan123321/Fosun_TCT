import torch
import numpy as np
from TCT_dataset import TCTDataset
from torch.utils.data import DataLoader
import time
import math
import os
os.environ["CUDA_VISIBLE_DEVICES"]=str(2)

import visdom
class Visualizer(object):
    def __init__(self,env='default',**kwargs):
        self.vis=visdom.Visdom(env=env,**kwargs)
        self.index={}
    def plot(self,name,y,**kwargs):
        x=self.index.get(name,0)
        self.vis.line(X=np.array([x]),Y=np.array([y]),win=name,opts=dict(title=name),update=None if x==0 else 'append',**kwargs)
        self.index[name]=x+1
    def txt(self,name,text,**kwargs):
        x=self.index.get(name,0)
        self.vis.text(text,win=name,append=False if x==0 else True,**kwargs)
        self.index[name]=x+1
vis=Visualizer("TCT_class_augment_balance_focial")


# step1: 模型
use_gpu=True
import pretrainedmodels

#print(pretrainedmodels.model_names)

#定义预训练模型
model_name="resnet18"
model=pretrainedmodels.__dict__[model_name](num_classes=1000,pretrained="imagenet")
#修改模型最后的输出类别
linear_TCT=torch.nn.Linear(in_features=512,out_features=2,bias=True)
model.last_linear=linear_TCT

if use_gpu:
    model.cuda()

# step2: 数据
from TCT_dataset import TCTDataset
batch_size=64
num_workers=8
#trSet = TCTDataset("/home/wangzehan/TCT_data/data_process/train", train=True)
trSet = TCTDataset("/home/wangzehan/TCT_data/data_augment/train",balance=True,train=True)
#valSet = TCTDataset("/home/wangzehan/TCT_data/data_process/val", train=False)
valSet = TCTDataset("/home/wangzehan/TCT_data/data_augment/val", train=False)
trainDataloader = DataLoader(trSet, batch_size,
                              shuffle=True,
                              num_workers=num_workers)
valDataloader = DataLoader(valSet, batch_size,
                            shuffle=False,
                            num_workers=num_workers)

# step3: 目标函数和优化器
focial_use=True
if focial_use:
    criterion=torch.nn.CrossEntropyLoss(reduction="none")
else:
    criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())#先使用默认学习率

# # 训练
max_epoch=100
use_gpu=True
for epoch_num in range(max_epoch):
    #为了与balance相匹配，因为dataloader不会随着epoch的改变而改变，这样可以丰富数据
    trainDataloader = DataLoader(trSet, batch_size,
                                 shuffle=True,
                                 num_workers=num_workers)

    avg_tr_loss = 0
    avg_tr_time = 0

    for ii, (image, label) in enumerate(trainDataloader):
        st_time = time.time()
        # 训练模型参数
        if use_gpu:
            image = image.cuda()
            label = label.cuda()
        optimizer.zero_grad()
        score = model(image)
        if focial_use:
            loss_cr = criterion(score, label)
            Pt = torch.exp(-loss_cr)
            loss = torch.mean((1 - Pt) ** 2 * loss_cr)
        else:
            loss = criterion(score, label)
        loss.backward()
        optimizer.step()
        batch_time = time.time() - st_time
        avg_tr_loss += loss.item()
        avg_tr_time += batch_time

        # 每100次训练记录一次损失
        if ii % 100 == 99:
            eta = avg_tr_time / 100 * (len(trSet) / batch_size - ii)
            print("Epoch no:", epoch_num + 1, "| Epoch progress(%):",
                  format(ii / (len(trSet) / batch_size) * 100, '0.2f'), "| Avg train loss:",
                  format(avg_tr_loss / 100, '0.4f'),"| ETA(s):", int(eta))
            text = "Epoch no:" + str(epoch_num + 1) + "| Epoch progress(%):" + str(
                ii / (len(trSet) / batch_size) * 100) + "| Avg train loss:" + str(
                avg_tr_loss / 100) + "| ETA(s):" + str(int(eta))
            vis.txt(name="record", text=text)
            vis.plot(name="loss", y=int(avg_tr_loss / 100))

    with torch.no_grad():
        print("Epoch",epoch_num+1,'complete. Calculating validation loss...')
        avg_val_loss = 0
        val_batch_count = 0
        total_points = 0

        for jj, (image, label) in enumerate(valDataloader):
            st_time = time.time()

            if use_gpu:
                image=image.cuda()
                label=label.cuda()

            # Forward pass
            score = model(image)
            if focial_use:
                loss_cr=criterion(score, label)
                Pt = torch.exp(-loss_cr)
                loss = torch.mean((1 - Pt) ** 2 * loss_cr)
            else:
                loss = criterion(score, label)

            avg_val_loss += loss.item()
            val_batch_count += 1

        print(avg_val_loss/val_batch_count)

        # Print validation loss and update display variables
        print('Validation loss :',format(avg_val_loss/val_batch_count,'0.4f'))
        textnew='Validation loss :'+str(avg_val_loss/val_batch_count)
        vis.txt(name="val",text=textnew)
        vis.plot(name="loss_val",y=int(avg_val_loss/val_batch_count))
        # val_loss.append(avg_val_loss/val_batch_count)
        # prev_val_loss = avg_val_loss/val_batch_count
    #__________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________
    torch.save(model.state_dict(), '../TCT_trainedmodels/resnet18_augment_balance_focial/cslstm_{}.tar'.format(str(epoch_num)))

