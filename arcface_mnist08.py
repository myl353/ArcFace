'''
网上的arcloss代码，
同时w和x都是二范数归一化，且s固定为具体值
arcloss(arcface)作为损失函数，类似center_loss那样，但与center_loss不同，
arcloss会将得到的输出也传入分类损失，得到另一个分类损失，与原来的分类损失相加得到最终的损失
而center_loss就是直接作为损失函数得到的损失与原分类损失相加得到最终的损失

两个优化器：分类-Adam，arcloss-Adam
与arcface06比，将网络改为网上那个了，分类层bias=False
在对特征输出加上了batchnorm
'''

import torch.nn as nn
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt
from torchvision.transforms import transforms
from PIL import Image,ImageDraw
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torch.optim as optim
import os

class ArcLoss(nn.Module):

    def __init__(self,feature_dim,cls_dim):
        super().__init__()
        #符合正态分布的方式初始化w
        self.weight = nn.Parameter(torch.randn(feature_dim,cls_dim))

        #s就是w的2范数归一化*x的2范数归一化，但arcface中w的摸的2范数归一化固定为1，x的2范数归一化固定为s
        #即s就是||w||*||x|| (注是2范数归一化操作，不是求模)
        self.s=10
        #m是加的角度
        self.m=0.1

    def forward(self, feature):
        #normalize不改变形状；norm(求模)会改变形状(将求模的轴的形状变为1，不给轴就是变为了常数)
        #求w和x的二范数归一化
        feature = F.normalize(feature, dim=1)
        # print("feature.shape:",feature.shape)
        w = F.normalize(self.weight, dim=0)
        # print("w.shape:",w.shape)

        #角度,除以10是为了防止梯度爆炸，公式中的cos_theat是x*w/(||x||*||w||)，而feature和w都是二范数归一化，也就是x/||x||和w/||w||
        #所以cos_theat就等于torch.matmul(feature,w)
        #如果feature和w前面只是求模，则cos_theat是torch.matmul(feature,w)/(torch.mm(feature,w))
        #注意只求模要矩阵相乘要注意形状(feature是N,w是C(类别数))，将两个变为N1和1C才能矩阵相乘
        cos_theat = torch.matmul(feature,w)/10
        #反三角函数反算角度
        a = torch.acos(cos_theat)

        #改变过的角度求的值，即分子 与 分母的第一个式子
        top = torch.exp(( torch.cos(a + self.m)) * self.s)
        #原来角度的值，要减去这个
        _top = torch.exp(( torch.cos(a)) * self.s)
        #分母的第二部分，要减去_top
        bottom = torch.sum(torch.exp(cos_theat * self.s), dim=1).view(-1, 1)

        ##公式中log后面的算式，只需要求这个就行了
        divide = (top / (bottom - _top + top)) + 1e-10

        '''
        #逻辑与上面是一样的，但试验效果有略微差异
        cos_theat = t.matmul(feature,w)/10
        sin_theat = t.sqrt(1.0-t.pow(cos_theat,2))
        cos_theat_m = cos_theat*t.cos(self.m)-sin_theat*t.sin(self.m)
        cos_theat_ = t.exp(cos_theat * self.s)
        sum_cos_theat = t.sum(t.exp(cos_theat*self.s),dim=1,keepdim=True)-cos_theat_
        top = t.exp(cos_theat_m*self.s)
        divide = (top/(top+sum_cos_theat))
        '''

        return divide

class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn_layer=nn.Sequential(
            nn.Conv2d(1, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.Conv2d(64, 256, 3, 2, 1),
            nn.BatchNorm2d(256),
            nn.PReLU(),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.PReLU(),
            nn.Conv2d(256, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.Conv2d(64, 16, 3, 2, 1),
            nn.BatchNorm2d(16),
            nn.PReLU()
        )

        self.feature_layer=nn.Sequential(
            nn.Linear(16*4*4,2),
            nn.BatchNorm1d(2),
            #nn.PRelu(),
            # nn.Linear(128,2),
        )

        self.out_layer=nn.Linear(2,10,bias=False)

    def forward(self, x):
        cnn_out=self.cnn_layer(x)
        # print(cnn_out.shape)

        x=torch.reshape(cnn_out,(-1,16*4*4))
        feature_out=self.feature_layer(x)

        out=F.log_softmax(self.out_layer(feature_out),dim=1)
        return feature_out,out


def draw_img(feature, targets, epoch, save_path='pics2'):
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
        plt.title("epoch={}".format(str(epoch)))

    plt.savefig('{}/{}.jpg'.format(save_path, epoch + 1))
    plt.draw()
    plt.pause(0.001)


if __name__ == '__main__':
    model_path='models/arcface_mnist07_.pth'
    arcloss_path = 'models/acrface_minist_arc07_.pth'

    #类别数
    class_num = 10



    net = Net().cuda()
    #导入主网络参数
    # if os.path.exists(model_path):
    #     net.load_state_dict(torch.load(model_path))

    arcloss_net = ArcLoss(2, 10).cuda()
    # # 导入arcloss的参数
    # if os.path.exists(arcloss_path):
    #     net.load_state_dict(torch.load(arcloss_path))

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    #导入mnist数据集
    dataset = MNIST('datasets/', train=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)


    #分类损失
    clsloss = nn.NLLLoss(reduction='sum').cuda()

    # # #分类的优化器
    # cls_opt=optim.SGD(net.parameters(),lr=0.0001,momentum=0.9,weight_decay=0.0005)
    # # #衰减系数，用来调整opt的学习率，
    # scheduler=optim.lr_scheduler.StepLR(cls_opt,step_size=20,gamma=0.8)
    cls_opt = optim.Adam(net.parameters())

    #arcloss的优化器
    arcloss_opt = optim.Adam(arcloss_net.parameters())


    i = 0
    while True:
        # 启用scheduler
        # scheduler.step()
        # 循环次数
        i += 1
        # 存储输出的特征点，即feature_out
        feat = []
        # 存储标签
        tar = []

        for j, (input, target) in enumerate(dataloader):
            input = input.cuda()
            target1 = target.cuda()
            target2 = torch.zeros(target.size(0), 10).scatter(1, target.view(-1, 1), 1).cuda()
            feature_out, out = net(input)

            value = torch.argmax(out, dim=1)

            arc_loss = torch.log(arcloss_net(feature_out))

            cls_loss = clsloss(out, target1)
            arcface_loss = clsloss(arc_loss, target1)

            loss = cls_loss + arcface_loss

            cls_opt.zero_grad()
            arcloss_opt.zero_grad()
            loss.backward()
            cls_opt.step()
            arcloss_opt.step()

            feat.append(feature_out)
            tar.append(target)

            print(f'epochs--{i}--{j}/{len(dataloader)},loss:{loss}')

        torch.save(net.state_dict(), model_path)
        torch.save(arcloss_net.state_dict(), arcloss_path)

        features = torch.cat(feat, dim=0)
        targets = torch.cat(tar, dim=0)
        draw_img(features.data.cpu(), targets.data.cpu(), i)