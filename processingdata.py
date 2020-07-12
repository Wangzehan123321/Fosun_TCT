import os
root_path="/home/wangzehan/TCT_data/smear2005/smear2005/New database pictures"
save_path="/home/wangzehan/TCT_data/data_process"
import random
from PIL import Image

for file_name in os.listdir(root_path):
    img_all_dir=[]
    path=os.path.join(root_path,file_name)
    for img in os.listdir(path):
        if img.split(".")[-1]=="BMP":
            img_all_dir.append(img)

    img_dir=[]
    for i in img_all_dir:
        img_dir.append(i[0:19])
    img_dir=list(set(img_dir))

    img_list=list(range(len(img_dir)))
    num_all=len(img_list)
    train_pick_num=int(len(img_list)*0.6)#6:2:2
    train_list=random.sample(img_list,train_pick_num)
    train_dir_list=[]
    for num in train_list:
        train_dir_list.append(img_dir[num])

    for image_name in img_all_dir:
        if image_name[0:19] in train_dir_list:
            image=Image.open(os.path.join(root_path,file_name,image_name))
            image.save(os.path.join(save_path,"train",image_name.split(".")[0]+"_"+str(file_name)+".BMP"))
    for num in train_list:
        img_list.remove(num)

    val_pick_num=int(len(img_list)*0.5)
    val_list=random.sample(img_list,val_pick_num)
    val_dir_list=[]
    for num in val_list:
        val_dir_list.append(img_dir[num])

    for image_name in img_all_dir:
        if image_name[0:19] in val_dir_list:
            image = Image.open(os.path.join(root_path, file_name, image_name))
            image.save(os.path.join(save_path, "val", image_name.split(".")[0] + "_" + str(file_name) + ".BMP"))
    for num in val_list:
        img_list.remove(num)

    test_list=img_list
    num_test=len(img_list)
    test_dir_list=[]
    for num in test_list:
        test_dir_list.append(img_dir[num])
    for image_name in img_all_dir:
        if image_name[0:19] in test_dir_list:
            image = Image.open(os.path.join(root_path, file_name, image_name))
            image.save(os.path.join(save_path, "test", image_name.split(".")[0] + "_" + str(file_name) + ".BMP"))
    assert num_test+val_pick_num+train_pick_num==num_all