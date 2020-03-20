'''
制作人脸数据库
'''

import torch
import numpy as np
from PIL import Image,ImageDraw
from torchvision.transforms import transforms
from arcface.arcface_face03 import Net
import os


net=Net().cuda()
net.load_state_dict(torch.load(r'models/arcface_face03.pth'))
net.eval()

feature_file=open('features.txt','w')

imgs_path=r'E:\Dataset\face3'
imgs_name=os.listdir(imgs_path)

for name in imgs_name:
    # print(name)
    img=Image.open(os.path.join(imgs_path,name))
    img=img.resize((100,100))
    img_tensor=transforms.Compose([
        transforms.ToTensor(), #CHW
        transforms.Normalize((0.6292, 0.5081, 0.4519),(0.2942, 0.2722, 0.2622))
        ])(img)
    img_tensor=img_tensor.unsqueeze(0).cuda()

    feature_out,out,feature_out2=net(img_tensor)
    #不能直接用item(),feature_out有多个数
    # print(feature_out.item())
    # print(feature_out.shape[1])

    # feature_out_1=[float(feature_out.data.cpu().numpy()[:,i]) for i in range(2)]
    # print(feature_out_1)


    #写入特征文件,这一句是把后面那个列表给写了进去
    #name.split('.')[0]是类别
    feature_file.write(f"{name.split('.')[0]}++{[float(feature_out.data.cpu().numpy()[:,i]) for i in range(feature_out.shape[1])]}\n")




