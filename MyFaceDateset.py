import torch
import numpy as np
from PIL import Image
from torchvision.transforms import transforms
from torch.utils.data import Dataset,DataLoader
import os

class MyFaceDataset(Dataset):
    def __init__(self,imgs_dir):
        self.imgs_dir=imgs_dir
        self.imgs_name=os.listdir(imgs_dir)


    def __len__(self):
        return len(self.imgs_name)

    def __getitem__(self, index):
        img_name=self.imgs_name[index]

        target=torch.tensor(int(img_name.split('.')[0]))
        # print(target)

        #WHC
        img=Image.open(os.path.join(self.imgs_dir,img_name))
        img=img.convert('RGB')
        img=img.resize((100,100))
        #HWC
        img=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.6183, 0.5017, 0.4506),(0.2953, 0.2725, 0.2616))
        ])(img)
        # print(img)

        return img,target


if __name__ == '__main__':
    imgs_dir=r'E:\Dataset\face2'
    dataset=MyFaceDataset(imgs_dir)
    # dataset[0]
    dataloader=DataLoader(dataset,batch_size=128,shuffle=True)

    # for i,(input,target) in enumerate(dataloader):
    #     print(input)
    #     print(target)
    #     break

    data=next(iter(dataloader))[0]
    mean = torch.mean(data, dim=(0, 2, 3))
    std = torch.std(data, dim=(0, 2, 3))
    print(mean)
    print(std)
