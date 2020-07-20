import torch
import os
from PIL import Image
root_dir="/home/wangzehan/TCT_data/data_process"
# criterion = torch.nn.CrossEntropyLoss(reduction="none")
# score=torch.tensor([[0.2,0.2,0.6],[0.2,0.2,0.6]])
# label=torch.tensor([2,2])
# print(criterion(score,label))
# import math
# pt=math.exp(0.6)/(math.exp(0.6)+2*math.exp(0.2))
# print(-math.log(pt)*(1-pt)*(1-pt))
#
# criterion = torch.nn.CrossEntropyLoss(reduction="none")
# Pt=torch.exp(-criterion(score,label))
# loss=torch.mean((1-Pt)**2*criterion(score,label))
# print(loss)

save_dir="/home/wangzehan/TCT_data/data_augment"
import torchvision.transforms as T
for file_name in os.listdir(root_dir):
    for img_name in os.listdir(os.path.join(root_dir,file_name)):
        img=Image.open(os.path.join(root_dir,file_name,img_name))
        for i in range(36):
            transform=T.Compose([T.RandomRotation(degrees=(10*i,10*i))])
            img_new=transform(img)
            img_name_new=img_name.split(".")[0]+"_"+str(10*i)+".BMP"
            img_new.save(os.path.join(save_dir,file_name,img_name_new))
