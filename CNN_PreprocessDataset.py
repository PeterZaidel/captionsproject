
# coding: utf-8

# In[1]:


import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from torch import optim
import torch
from torch import nn
from torch.autograd import Variable
import pandas
from sklearn.preprocessing import MinMaxScaler
import torchvision.models as models
import torchvision.transforms as transforms
import torch
from torch.autograd import Variable
import numpy as np
from PIL import Image
import copy
import shutil

from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
pylab.rcParams['figure.figsize'] = (8.0, 10.0)
from torch.utils.data import Dataset, DataLoader

from dataset import MSCOCODataset
from torch.optim import lr_scheduler
from autocorrect import spell
import nltk
from IPython.display import display
import os

from tqdm import tqdm
from tqdm import tqdm_notebook


# In[2]:


DEF_SEND = '<SEND>'
DEF_START = '<START>'


# In[3]:


TRAIN_DATSET_FILE = 'traindataset_cnn.tar.gz'
TEST_DATSET_FILE = 'testdataset_cnn.tar.gz'


# In[4]:


dataDir='/home/p.zaydel/ProjectNeuralNets/coco_dataset/'
imagesDirTrain = '{}train2017/train2017'.format(dataDir)
imagesDirVal = '{}val2017/val2017'.format(dataDir)

annTrainFile = '{}/annotations_trainval2017/annotations/captions_train2017.json'.format(dataDir)
annValFile = '{}/annotations_trainval2017/annotations/captions_val2017.json'.format(dataDir)


# In[5]:


transform_tensor = transforms.Compose([
                                transforms.ToTensor(), 
                                transforms.Normalize(
                                    mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
                                           ])
transform_to224 = transforms.Compose([transforms.Resize((224, 224)),
                                      transform_tensor
                                     ])
transform_to500 = transforms.Compose([ transforms.Resize((500, 500)),
                                      transform_tensor
                                           ])


# In[6]:



def split_text2words(text):
    symbs_to_replace = ['.', ',', '/', '-', ':', '{', '}', '[', ']', ]
    for smb in symbs_to_replace:
        text = text.replace(smb, ' ')
    
    
    words = nltk.word_tokenize(text.lower())
    
    for idx in range(len(words)):
        words[idx] = spell(words[idx])
    
    words = [DEF_START] + words + [DEF_SEND]
    
    return words

# def anns2words(anns_list):
#     texts = []
#     for anns in anns_list:
#         for ann in anns['anns']:
#             words = split_text2words(ann)
#             texts.append(words)
            
#     return texts

from gensim.models import Word2Vec
def train_word_to_vec_gensim(dataset, embed_size = 300):
    Texts = dataset.anns.values()
    model = Word2Vec(Texts, size = embed_size)
    return model

def generate_vocab_dicts(dataset): 
    Texts = dataset.anns.values()
    uniqwords = list(set([w for ann in Texts for w in ann]))
    words2ids = dict(zip(uniqwords, range(len(uniqwords))) )
    ids2words = dict(zip(range(len(uniqwords)), uniqwords ))
    return words2ids, ids2words


def wordslist2wordids(words, word2id, vector_length = None ):
    if vector_length is None:
        word_ids = [word2id[w] for w in words]
        
    else:
        word_ids = []
        for idx in range(vector_length):
            if idx < len(words):
                w = words[idx]
            else:
                w = end_word
                
            word_ids.append(word2id[w])
        
        if word_ids[-1] != word2id[DEF_SEND]:
            word_ids[-1] = word2id[DEF_SEND]
        
    return torch.from_numpy(np.array(word_ids).astype(np.int))


def sentence2wordids(sentence, word2id, vector_length = None):
    
    if vector_length is None:
        words = split_text2words(sentence)
        word_ids = [word2id[w] for w in words]
        
    else:
        words = split_text2words(sentence)
        word_ids = []
        for idx in range(vector_length):
            if idx < len(words):
                w = words[idx]
            else:
                w = end_word
                
            word_ids.append(word2id[w])
        
        if word_ids[-1] != word2id[DEF_SEND]:
            word_ids[-1] = word2id[DEF_SEND]
        
    return torch.from_numpy(np.array(word_ids).astype(np.int))         
    
    
import numpy as np
# calculates dimension of alexnet convolutions layers output 
def get_alexnet_features_dim(imsize):
    adim = int(np.round( 3*0.01*imsize - 1))
    return 1*256*adim*adim


# In[23]:


def save_prepared_dataset(dataset, filename, cnn_model = models.alexnet(pretrained=True).features):
    
    print("preparing images...")
    for idx in tqdm(range(len(dataset))):
        sample = dataset[idx]
        dataset.images_cnn[idx] =  cnn_model(Variable(sample['image'].unsqueeze(0))).data.view(-1)
        
    print("preparing annotations...")
    dataset.text_transform = split_text2words
    dataset.preload_anotations()
    dataset.text_transform = None
    
    torch.save(dataset, filename)
    print("Dataset saved in {}".format(filename))


# In[24]:


def load_anns(dataset, annids, max_len, prepare = None):
    '''
       dataset - MSCOCODataset
       annids -  tensor or numpy array
       max_len - maximum len of sentence. If None computes from dataset 
       prepare - None or function to prepare each word, returns 1-dim tensor
       
       return Pytorch Tensor [len(annids) x max_sentence_len x prepare(word).shape[0] ]
    '''
    result = []
    
    if prepare is None:
        prepare = lambda w: word_embeding[w]
    
    for i in range(annids.shape[0]):
        words = dataset.get_ann(annids[i])
        ann_res = []
        
        for idx in range(max_len):
            if idx < len(words):
                w = words[idx]
            else:
                w = DEF_SEND
                
            ann_res.append(prepare(w))
        ann_res = torch.from_numpy(np.array(ann_res)).float()
        result.append(ann_res)
        
    return torch.stack(result)


# In[25]:


import os
print("Creating dictionary......")
if os.path.exists('dictionaries.tar.gz'):
    print("loading dictionary")
    dic_state = torch.load('dictionaries.tar.gz')
    words2ids = dic_state['words2ids']
    ids2words = dic_state['ids2words']
    print("dictionary loaded")
else:
    words2ids, ids2words  = generate_vocab_dicts(trainDataset)
    print("saving dictionary")
    torch.save({'words2ids': words2ids, 'ids2words': ids2words }, 'dictionaries.tar.gz')


# In[26]:


# MY WORD EMBEDDINGS
import os

WORD_EMBED_FILE = 'word_embeding.tar.gz'
if os.path.exists(WORD_EMBED_FILE):
    print("loading words embedding")
    word_embeding = torch.load(WORD_EMBED_FILE)
    print("words embedding loaded")
else:
    print("creating words embedding......")
    word_embeding = train_word_to_vec_gensim(trainDataset, embed_size = 300 )
    print("saving words embedding")
    torch.save(word_embeding, WORD_EMBED_FILE)


# In[27]:

cnn_model = models.alexnet(pretrained=True).features
if os.path.exists(TRAIN_DATSET_FILE):
    print("loading train dataset...")
    trainDataset = torch.load(TRAIN_DATSET_FILE)
    print('train dataset loaded!')
else:
    trainDataset = MSCOCODataset(annTrainFile,imagesDirTrain, transform = transform_to224, mode='pic2rand')
    save_prepared_dataset(trainDataset, TRAIN_DATSET_FILE)
    
if os.path.exists(TEST_DATSET_FILE):
    print("loading test dataset...")
    testDataset = torch.load(TEST_DATSET_FILE)
    print('test dataset loaded!')
else:
    testDataset = MSCOCODataset(annValFile,imagesDirVal, transform = transform_to224, mode='pic2rand')
    save_prepared_dataset(testDataset, TEST_DATSET_FILE)

print("SUCCESS!!!!!")
# In[12]:


# np.stack([trainDataset[100]['image'][0].numpy(), 
#              trainDataset[100]['image'][0].numpy(), 
#              trainDataset[100]['image'][0].numpy() ]).shape


# In[21]:


# cnn_model = models.alexnet(pretrained=True).features


# In[18]:


# Variable(trainDataset[101]['image'])


# In[19]:


# trainDataset[101]['image']


# In[20]:


# cnn_model(Variable(trainDataset[0]['image'].unsqueeze(0))).view(-1).data

