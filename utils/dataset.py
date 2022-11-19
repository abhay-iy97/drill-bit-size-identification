import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from utils.logger import NativeLogger

class DrillDataset(Dataset):
    def __init__(self, imgPath):
        """Drill dataset constructor - initializes all data related to train/test/val

        Args:
            imgPath (string): Location to folder structure which contains processed frames
        """
        self.img_path = imgPath
        fileList = glob.glob(self.img_path + '*')
        self.dataList = []
        for classDir in fileList:
            className = classDir.split('/')[-1]
            for imgFilePath in glob.glob(classDir + '/*.jpg'):
                self.dataList.append([imgFilePath, className])
            # print(self.dataList)
        self.classMap = {'20x26dark': 0, '20x26light': 0, '20x28dark': 1,  '20x28light': 1,  '28x22dark': 2,  '28x22light': 2,  '35x19dark': 3,  '35x19light': 3,  '35x22dark': 4,  '35x22light': 4,  '35x28dark': 5,  '35x28light': 5, '35x30dark': 6,  '35x30light': 6,  '42x22dark': 7,  '42x22light': 7,  '42x30dark': 8,  '42x30light': 8}
        self.classNames = {0: '2.0 mm x 26 mm', 1: '2.0 mm x 28 mm', 2: '2.8 mm x 22 mm', 3: '3.5 mm x 19 mm', 4: '3.5 mm x 22 mm', 5: '3.5 mm x 28 mm', 6: '3.5 mm x 30 mm', 7: '4.2 mm x 22 mm', 8: '4.2 mm x 30 mm'}
        self.imgDim = (1280, 960)

    
    def __len__(self):
        """Return length of the dataloader

        Returns:
            int: Length of the dataloader
        """
        return len(self.dataList)
    
    def __getitem__(self, idx):
        """Get an element from the dataloader.

        Args:
            idx (int): index

        Returns:
            torch.tensor: Image tensor object
            torch.tensor: Class Id such as 0, 1, 2 etc. These are mapped to the actual class names above in self.classNames
        """
        imgPath, className = self.dataList[idx]
        img = cv2.imread(imgPath)
        img = cv2.resize(img, self.imgDim)
        classId = torch.tensor(self.classMap.get(className))
        imgTensor = torch.from_numpy(img)
        imgTensor = imgTensor.permute(2, 0, 1)
        return imgTensor, classId
    
    def getClassNames(self):
        """Return dictionary of class mames

        Returns:
            dict: Class name dictionary
        """
        return self.classNames


