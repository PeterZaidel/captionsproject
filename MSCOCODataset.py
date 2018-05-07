#!/usr/bin/env python

import numpy as np
import torch
from torch import nn
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from pycocotools.coco import COCO
import skimage.io as io
from torch.utils.data import Dataset, DataLoader

import tqdm

def numpy2image(img_numpy):
    if img_numpy.dtype == np.dtype('float64'): #нужно домножить на 255 и поменять тип
        img_numpy = (img_numpy*255).astype('uint8')
    return Image.fromarray(img_numpy)

class MSCOCODataset(Dataset):
    """MSCOCO Dataset"""

    

    def __init__(self, annFile, imagesDir, preload = False, transform = None):
        self.transform = transform
        self.coco = COCO(annFile)
        self.imagesDir = imagesDir
        self.imageids = self.coco.getImgIds()
        self._data_preload = []
        
        
        if preload == True:
            self.__preload()
    
    def __preload(self):
        for i in tqdm.tqdm_notebook(range(len(self.imageids))):
            imid = self.imageids[i]
            img_data = self.coco.loadImgs([imid])[0]
        
            annIds = self.coco.getAnnIds(imgIds=imid)
        
            anns_data = self.coco.loadAnns(annIds)
            anns = [ann['caption'] for ann in anns_data]
        
        
            img_file_name = '{}/{}'.format(self.imagesDir, img_data['file_name'])
            image = numpy2image(io.imread(img_file_name))
            if self.transform:
                image = self.transform(image)
            
            sample = {'id': imid, 'image': image, 'anns': anns}
                
            self._data_preload.append(sample)


    def __len__(self):
        return len(self.coco.dataset['images'])

    def __getitem__(self, idx):
        
        if self.preload == True:
            return self._data_preload[idx]
        
        
        imid = self.imageids[idx]
        
        img_data = self.coco.loadImgs([imid])[0]
        
        annIds = self.coco.getAnnIds(imgIds=imid)
        
        anns_data = self.coco.loadAnns(annIds)
        anns = [ann['caption'] for ann in anns_data]
        
        
        img_file_name = '{}/{}'.format(self.imagesDir, img_data['file_name'])
        image = numpy2image(io.imread(img_file_name))
        if self.transform:
            image = self.transform(image)
        
        sample = {'id': imid, 'image': image, 'anns': anns}

        return sample

