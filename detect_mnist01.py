# -*- coding: utf-8 -*-
'''
是使用gpu来跑
'''

import torch
from arcface.arcface_mnist06 import Net
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import matplotlib.pyplot as plt
import os


def draw_img(feature, targets, save_path='pics1'):
    if os.path.isdir(save_path) != True:
        os.makedirs(save_path)

    color = ["red", "black", "yellow", "green", "pink", "gray", "lightgreen", "orange", "blue", "teal"]
    cls = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    plt.ion()
    plt.clf()

    for j in cls:
        #mask如果targets == j就为True
        mask = [targets == j]
        #找到类别为j的特征点
        feature_ = feature[mask].numpy()
        x = feature_[:, 1]
        y = feature_[:, 0]
        label = cls
        plt.plot(x, y, ".", color=color[j])
        plt.legend(label, loc="upper right")  # 如果写在plot上面，则标签内容不能显示完整
        plt.title("test")

    plt.savefig('{}/test.jpg'.format(save_path))
    plt.draw()
    plt.pause(0.001)



if __name__ == '__main__':

    net=Net().cuda()
    net.load_state_dict(torch.load('models/arcface_mnist06.pth'))

    net.eval()

    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    test_data=MNIST('datasets/',train=False,transform=transform)
    test_dataloader=DataLoader(test_data,batch_size=128,shuffle=True)

    # 存储输出的特征点，即feature_out
    feat = []
    # 存储标签
    tar = []

    total=0.
    cerrect=0.

    with torch.no_grad():
        for i , (input,target) in enumerate(test_dataloader):
            input = input.cuda()
            target1 = target.cuda()

            feature_out, out = net(input)

            feat.append(feature_out)

            #因为要看实际类别的分布情况，所以这里使用的是真实类别，而不是网络的判断类别
            tar.append(target)

            total += target.size(0)
            cerrect += (torch.argmax(out,dim=1) == target1).sum()




            del i ,input,target,feature_out,out
        acc = float(cerrect) / total
        print(acc)

        features = torch.cat(feat, dim=0)
        targets = torch.cat(tar, dim=0)
        draw_img(features.data.cpu(), targets.data.cpu())










