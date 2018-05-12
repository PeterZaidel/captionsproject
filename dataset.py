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
import random
from autocorrect import spell
import os

import tqdm

def numpy2image(img_numpy):
    if img_numpy.dtype == np.dtype('float64'): #нужно домножить на 255 и поменять тип
        img_numpy = (img_numpy*255).astype('uint8')
    return Image.fromarray(img_numpy)



PIC2MANY = 'pic2many'
PIC2RAND = 'pic2rand'
ANN2PIC = 'ann2pic'

class MSCOCODataset(Dataset):
    """MSCOCO Dataset"""
    
    def get_image2anns(self):
        result = []
        for imid in self.imageids:
            annIds = self.coco.getAnnIds(imgIds=imid)
        
            anns_data = self.coco.loadAnns(annIds)
            anns = [ann['caption'] for ann in anns_data]
            result.append({'id': imid, 'anns': anns})
            
        return result
    
    def get_anns(self):
        result = {}
        for annid in tqdm.tqdm_notebook(self.annids):
#             ann = self.coco.loadAnns([annid])[0]['caption']
#             if self.text_transform:
#                 ann = self.text_transform(ann)
            result[annid] = self.get_ann(annid)
            
        return result
        
        #preload = False,
    def __init__(self, annFile, imagesDir, transform = None, 
                 mode = 'pic2many', text_transform = None):
        
        self.transform = transform
        self.text_transform = text_transform
        
        self.coco = COCO(annFile)
        
        self.imagesDir = imagesDir
        self.imageids = self.coco.getImgIds()
        self.annids = self.coco.getAnnIds()
        
        self.preload = False
        self.preload_anns = False
        
        #self._data_preload = []
        
        
        self.mode = mode
        
#         if preload == True:
#             self.__preload()
#         self.preload = preload
        
        self.anns = {}
        
        
    def preload_anotations(self):
        self.anns = self.get_anns()
        self.preload_anns = True
        
    
    def preload_data(self):
        self.preload = True
        for sample in tqdm.tqdm_notebook(self):        
            self._data_preload.append(sample)
#             if self.mode == PIC2MANY or self.mode == PIC2RAND:
#             for i in tqdm.tqdm_notebook(range(len(self.imageids))):
#                 imid = self.imageids[i]
#                 img_data = self.coco.loadImgs([imid])[0]
        
#                 annIds = self.coco.getAnnIds(imgIds=imid)
        
#                 anns_data = self.coco.loadAnns(annIds)
#                 anns = [ann['caption'] for ann in anns_data]
        
        
#                 img_file_name = '{}/{}'.format(self.imagesDir, img_data['file_name'])
#                 image = numpy2image(io.imread(img_file_name))
#                 if self.transform:
#                     image = self.transform(image)
            
#                 sample = {'id': imid, 'image': image, 'anns': anns}
                
#                 self._data_preload.append(sample)
        

    def __len__(self):
        if self.mode == PIC2MANY:
            return len(self.coco.dataset['images'])
        elif self.mode == PIC2RAND:
            return len(self.coco.dataset['images'])
        elif self.mode == ANN2PIC:
            return len(self.coco.dataset['annotations'])
    
    def get_item_data(self, idx):
        
        if self.mode == PIC2MANY or self.mode == PIC2RAND:        
            imid = self.imageids[idx]
            img_data = self.coco.loadImgs([imid])[0]
        
            annIds = self.coco.getAnnIds(imgIds=imid)
        
            anns_data = self.coco.loadAnns(annIds)
            anns = [ann['caption'] for ann in anns_data]
            if self.mode == PIC2RAND:
                anns = anns[random.randint(0,len(anns) - 1 )]
        
            img_file_name = '{}/{}'.format(self.imagesDir, img_data['file_name'])
      
            sample = {'id': imid,'anns_ids': annIds, 'img_data': img_data, 
                  'image_file': img_file_name, 'anns': anns}

            return sample
            
        elif self.mode == ANN2PIC:
            annid = self.annids[idx]
            ann_data = self.coco.loadAnns([annid])[0]
            anns = ann_data['caption']
            imid = ann_data['image_id']
            img_data = self.coco.loadImgs([imid])[0]
            
            img_file_name = '{}/{}'.format(self.imagesDir, img_data['file_name'])
      
            sample = {'id': imid, 'anns_ids': annid,'img_data': img_data, 
                  'image_file': img_file_name, 'anns': anns}

            return sample
        
        
    
    def get_ann(self, annid):
        
        if self.preload_anns == True:
            return self.anns[annid]
        
        ann = self.coco.loadAnns([annid])[0]['caption']
        if self.text_transform:
            ann = self.text_transform(ann)
        return ann
    
#         if self.text_transform:
#             if self.mode == PIC2MANY:
#                 for idx in range(len(anns)):
#                     anns[idx] = self.text_transform(anns[idx])
                    
#             elif self.mode == PIC2RAND or self.mode == ANN2PIC:
#                 anns = self.text_transform(anns)
       

    def __getitem__(self, idx):
        
        if self.preload == True:
            return self._data_preload[idx]
        
        
        item_data = self.get_item_data(idx)
        
        img_file_name = item_data['image_file']
        imid = item_data['id']
        annids = item_data['anns_ids']
        #anns = item_data['anns']
        
        image = io.imread(img_file_name)
        
        # for black-white images
        if len(image.shape) != 3:
            return self.__getitem__(0)
        
        image = numpy2image(image)
        if self.transform:
            image = self.transform(image)
                
        
        #sample = {'imid': imid, 'image': image, 'anns': anns}
        if self.mode == ANN2PIC or self.mode == PIC2RAND:
            sample = {'image': image, 'anns': annids[0]}
        else:
            sample = {'image': image, 'anns': annids}

        return sample
            
        
        
#         if self.mode == PIC2MANY or self.mode == PIC2RAND:
        
#             imid = self.imageids[idx]
        
#             img_data = self.coco.loadImgs([imid])[0]
        
#             annIds = self.coco.getAnnIds(imgIds=imid)
        
#             anns_data = self.coco.loadAnns(annIds)
#             anns = [ann['caption'] for ann in anns_data]
#             if self.mode == PIC2RAND:
#                 anns = anns[random.randint(0,len(anns))]
        
        
#             img_file_name = '{}/{}'.format(self.imagesDir, img_data['file_name'])
#             image = numpy2image(io.imread(img_file_name))
#             if self.transform:
#                 image = self.transform(image)
        
#             sample = {'id': imid, 'image': image, 'anns': anns}

#             return sample
        
#         elif self.mode == ANN2PIC:
            
#             annid = self.annids[idx]
#             ann_data = self.coco.loadAnns([annid])
            
#             anns = ann_data['caption']
#             imid = ann_data['image_id']
#             img_data = self.coco.loadImgs([imid])[0]
#             img_file_name = '{}/{}'.format(self.imagesDir, img_data['file_name'])
            
#             image = numpy2image(io.imread(img_file_name))
#             if self.transform:
#                 image = self.transform(image)
        
#             sample = {'id': imid, 'image': image, 'anns': anns}

#             return sample
            
                
        

