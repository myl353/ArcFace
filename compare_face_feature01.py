'''
与make_face_database 的编号(后面的01)对应，因为写入文件的方式和读取文件的方式对应
使用，与人脸数据库中的特征做对比
'''

import torch
import numpy as np
from PIL import Image,ImageDraw
from torchvision.transforms import transforms
from arcface.arcface_face03 import Net
import os


def CosSimilarity(feature_out):
    feature_file = open('features.txt')
    for line in feature_file:
        cls,feature=line.split('++')
        # print(feature)

        #去掉前面的[ 和后面的 ]与\n,再以 , 分割
        feature_list=feature[1:-2].split(',')
        # print(feature_list)
        fea_arr=np.array([float(feature_list[i]) for i in range(len(feature_list))])
        # print(fea_arr)

        #如果不在这里加维，就要在feature_out的部分减去维度，保证两个向量维度相同，且计算sim时，维度就要改为0
        fea_tensor=torch.Tensor(fea_arr).unsqueeze(0)
        # print(fea_tensor.shape)

        sim=torch.cosine_similarity(fea_tensor,feature_out,dim=1)
        # print(sim)

        if sim > 0.9:
            print(f'ta是{cls_[cls]}')

    return

if __name__ == '__main__':

    net=Net().cuda()
    net.load_state_dict(torch.load(r'models/arcface_face03.pth'))
    net.eval()

    cls_={'0':'刘德华','1':'刘诗诗','2':'迪丽热巴','3':'陈乔恩','4':'成龙'}

    imgs_path=r'E:\Dataset\face4'
    imgs_name=os.listdir(imgs_path)

    for name in imgs_name:
        print(name)
        img=Image.open(os.path.join(imgs_path,name))
        img=img.resize((100,100))
        img_tensor=transforms.Compose([
            transforms.ToTensor(), #CHW
            transforms.Normalize((0.6292, 0.5081, 0.4519),(0.2942, 0.2722, 0.2622))
            ])(img)
        img_tensor=img_tensor.unsqueeze(0).cuda()

        feature_out,out,feature_out2=net(img_tensor)
        # print(feature_out.shape)
        feature_out=feature_out.cpu().data

        sim=CosSimilarity(feature_out)
        # print(sim)
        # break



