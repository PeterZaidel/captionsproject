
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
from torch.optim import lr_scheduler
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

import nltk
from IPython.display import display


# In[2]:


gpu_device = 3


# In[3]:


dataDir='/home/p.zaydel/ProjectNeuralNets/coco_dataset/'
imagesDirTrain = '{}train2017/train2017'.format(dataDir)
imagesDirVal = '{}val2017/val2017'.format(dataDir)

annTrainFile = '{}/annotations_trainval2017/annotations/captions_train2017.json'.format(dataDir)
annValFile = '{}/annotations_trainval2017/annotations/captions_val2017.json'.format(dataDir)


# In[4]:


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


# In[5]:


trainDataset = MSCOCODataset(annTrainFile,imagesDirTrain, transform = transform_to500, mode='pic2rand')
testDataset = MSCOCODataset(annValFile,imagesDirVal, transform = transform_to500, mode='pic2rand')


# In[6]:


# idx = 0
# print(trainDataset[idx]['anns'])
# print(trainDataset[idx]['imid'])
# transforms.ToPILImage()(trainDataset[idx]['image'])


# In[7]:


DEF_SEND = 'SEND'
def split_text2words(text, end_word = DEF_SEND):
    words = nltk.word_tokenize(text.lower())
    if words[-1] == '.':
        words.pop(-1)
        
    words.append(end_word)
    return words

def anns2words(anns_list, end_word = DEF_SEND):
    texts = []
    for anns in anns_list:
        for ann in anns['anns']:
            words = split_text2words(ann, end_word)
            texts.append(words)
            
    return texts

from gensim.models import Word2Vec
def train_word_to_vec_gensim(dataset, embed_size = 4096, end_word = DEF_SEND):
    Anns = dataset.get_anns()
    Texts = anns2words(Anns, end_word)
    model = Word2Vec(Texts, size = embed_size)
    return model

def generate_vocab_dicts(dataset, end_word = DEF_SEND):
    Anns = dataset.get_anns()
    Texts = anns2words(Anns, end_word)
    uniqwords = list(set([w for ann in Texts for w in ann]))
    words2ids = dict(zip(uniqwords, range(len(uniqwords))) )
    ids2words = dict(zip(range(len(uniqwords)), uniqwords ))
    return words2ids, ids2words

def sentence2wordids(sentence, word2id, vector_length = None, end_word = DEF_SEND):
    
    if vector_length is None:
        words = split_text2words(sentence, end_word)
        word_ids = [word2id[w] for w in words]
        
    else:
        words = split_text2words(sentence, end_word)
        word_ids = []
        for idx in range(vector_length):
            if idx < len(words):
                w = words[idx]
            else:
                w = end_word
                
            word_ids.append(word2id[w])
        
        if word_ids[-1] != word2id[end_word]:
            word_ids[-1] = word2id[end_word]
        
    return torch.from_numpy(np.array(word_ids).astype(np.int))         
    
    
import numpy as np
# calculates dimension of alexnet convolutions layers output 
def get_alexnet_features_dim(imsize):
    adim = int(np.round( 3*0.01*imsize - 1))
    return 1*256*adim*adim


# In[8]:


# trainAnnCaps = [ann['caption'] for ann in trainDataset.coco.loadAnns(trainDataset.coco.getAnnIds())]

# trainAnns = trainDataset.get_anns()
# trainTexts = anns2words(trainAnns)
# sent_lengths = np.array([len(ann) for ann in trainTexts])
# print("max sent id", sent_lengths.argmax())
# print('max len',np.max(sent_lengths))
# plt.plot(np.unique(sent_lengths), np.bincount(sent_lengths)[6:])


# In[9]:


# testAnns = testDataset.get_anns()
# testTexts = anns2words(testAnns)
# test_sent_lengths = np.array([len(ann) for ann in testTexts])
# print("max sent id", test_sent_lengths.argmax())
# print('max len',np.max(test_sent_lengths))
# test_bin_count = np.bincount(test_sent_lengths)
# plt.plot(np.unique(test_sent_lengths), test_bin_count[test_bin_count > 0])


# In[10]:


# Anns = trainDataset.get_anns()
# Texts = anns2words(Anns)


# In[11]:

print("Creating dictionary......")
words2ids, ids2words  = generate_vocab_dicts(trainDataset)


# In[12]:


#word_embeding = train_word_to_vec_gensim(trainDataset, embed_size = 4096 )


# In[13]:


#sentence2wordids(Anns[0]['anns'][3], words2ids,  vector_length = 20 )


# In[14]:


text_transform = lambda text: sentence2wordids(text, words2ids, vector_length = 20)
trainDataset.text_transform = text_transform
testDataset.text_transform = text_transform


# In[15]:


trainDataLoader = DataLoader(trainDataset, batch_size = 64, shuffle=True)
testDataLoader = DataLoader(testDataset, batch_size = 64, shuffle=True)


# In[16]:


# for sample in trainDataLoader:
#     break
# sample


# In[17]:


# vec = word_embeding['.']
# print(vec)
# word_embeding.wv.similar_by_vector(vec)


# In[18]:


# len(words2ids)


# In[31]:


class LSTM_W2V_Net(nn.Module):

    def __init__(self,  image_size, image_features_size, word_embedding, 
                 word_embedding_size, words2ids, ids2words,
                 
                 cnn = models.alexnet(pretrained=True).features, 
              #   cnn_comp_features = lambda cnn, x: cnn.features(x),
                 max_sentence_len = 20,
                 sentence_end_embed = None,
                 sentence_end_symbol = '.'
                  ):
        """Init NN
            image_size - size of input image.
            hidden_size - size of cnn features output
            word_embedding - pretrained model wor word embedding
            word_embedding_size - dimension of embedding space
            words2ids - dictionary word -> id
            ids2words - dictionary id -> word
            cnn - pretrained cnn net (alexnet, vgg and other)
            cnn_comp_features - function computes features with cnn
            max_sentence_len - maximum sentence length when lstm stops
        """
        
        super(LSTM_W2V_Net, self).__init__()
        self.image_size = image_size
        self.image_features_size = image_features_size
        self.cnn = cnn 
     #   self.cnn_comp_features = cnn_comp_features
        
        self.vocab_size = len(words2ids)
        self.word_embedding_size = word_embedding_size
        #self.words_embedding = word_embedding
        
        self.words2ids = words2ids
        self.ids2words = ids2words
        
#         self.sentence_end_symbol = sentence_end_symbol
#         self.sentence_end_symbol_id = self.words2ids[self.sentence_end_symbol]
        
#         if sentence_end_embed is not None:
#             self.sentence_end_embed = sentence_end_embed
#         else:
#             self.sentence_end_embed = word_embeding['.']
        
        self.max_sentence_len = max_sentence_len
        self.hidden_size = word_embedding_size 
        self.fc1 = nn.Sequential( nn.BatchNorm1d(self.image_features_size),
                                  nn.Linear(self.image_features_size, int(self.image_features_size/2)),
                                  nn.ReLU(),
                                  nn.Dropout(),
                                  nn.Linear(int(self.image_features_size/2), int(self.image_features_size/4) ),
                                  nn.ReLU(),
                                  nn.Dropout(),
                                  nn.Linear(int(self.image_features_size/4), self.hidden_size),
                                  nn.BatchNorm1d(self.hidden_size)
                                ) 
        
        self.fc2 = nn.Sequential(nn.Linear(self.hidden_size, self.vocab_size )
                                  ,nn.LogSoftmax()
                                ) 
        
                               
        self.lstm_cell = nn.LSTMCell(self.hidden_size, self.hidden_size) 
        #self.lstm = nn.LSTM(hidden_size, word_embedding_size)
    
        
    
    def freeze_cnn(self):
        for param in self.cnn.parameters():
            param.requires_grad = False
    
    def unfreeze_cnn(self):
        for param in self.cnn.parameters():
            param.requires_grad = True
    
    def forward(self, X):
        # get features from images
        batch_size = X.shape[0]
        #print("1: " ,X.shape)
        X = self.cnn(X)
        #X = X 
        
        #print("2: ",X.shape)
        X = X.view(batch_size, self.image_features_size)
        
    
        h_t = Variable(torch.zeros(batch_size, self.hidden_size), requires_grad=False) 
        c_t = Variable(torch.zeros(batch_size, self.hidden_size), requires_grad=False) 
        
        X = self.fc1.forward(X)
        
        h_t, c_t = self.lstm_cell.forward(X, (h_t, c_t))
        
        output = []
        for idx in range(self.max_sentence_len):
            h_t, c_t = self.lstm_cell.forward(X, (h_t, c_t))
            
            r = self.fc2.forward(h_t)
            
            #logits = nn.LogSoftmax(r).max(2)[1]
            
            output.append(r)
        
        output = torch.stack(output, 1) 
        return output


# In[32]:


image_size = 500
image_features_size = get_alexnet_features_dim(image_size)
word_embeding_size = 1024#word_embeding.trainables.layer1_size
sentence_end_embed = 1#word_embeding[DEF_SEND]
cnn = models.alexnet(pretrained=True).features
sentence_end_symbol = DEF_SEND
max_sentence_len = 20


# In[33]:

print("Initializing LSTM......")
lstmnet = LSTM_W2V_Net(image_size, image_features_size , None,
                     word_embeding_size, words2ids, ids2words, 
                     cnn = cnn,
                     max_sentence_len = max_sentence_len,
                     sentence_end_embed = sentence_end_embed)


# In[34]:

print("Testing LSTM......")
trainDataLoader_2 = DataLoader(trainDataset, batch_size = 2, shuffle=True)
for sample in trainDataLoader_2:
    break


# In[35]:


optimizer = torch.optim.Adam(lstmnet.parameters(), lr=0.001)
optimizer.zero_grad()


# In[36]:


lstmnet.freeze_cnn()
X = Variable(sample['image']) 
pred = lstmnet.forward(X)
y = sample['anns']
y = Variable(y) 
loss = nn.NLLLoss2d()(pred.view(pred.shape[0]*pred.shape[1], pred.shape[2]), y.view(-1))


loss.backward()
optimizer.step()
print("TEST OK!")



def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'best_'+filename)
        
def open_checkpoint(is_best = False, filename='checkpoint.pth.tar'):
    if is_best:
        filename = 'best_'+filename
        
    checkpoint = torch.load(filename)
    return checkpoint
#     best_prec1 = checkpoint['best_prec1']
#     model.load_state_dict(checkpoint['state_dict'])
#     optimizer.load_state_dict(checkpoint['optimizer'])


# In[149]:


train(lstmnet, trainDataLoader, testDataLoader, 20)



def train(network, train_dataloader, test_dataloader,
          epochs, unfreeze_cnn_epoch = None,
          loss = nn.NLLLoss() , optim=torch.optim.Adam ):

    print("Train Started!")
    if unfreeze_cnn_epoch is None:
        unfreeze_cnn_epoch = int(0.75 * epochs)
    
    train_loss_epochs = []
    test_loss_epochs = []
    optimizer = optim(network.parameters(), lr=0.001)
    best_test_score = 10**6
    
    sheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)
    network.freeze_cnn()
    
    try:
        for epoch in range(epochs):
            sheduler.step()
            if epoch >= unfreeze_cnn_epoch:
                network.unfreeze_cnn()

            losses = []
            accuracies = []
            for sample in train_dataloader:
                X = sample['image']
                X = Variable(X)
                y = sample['anns']
                
                # одно изображение - одно предложение
                
                y = Variable(y)
                
                
                prediction = network(X)
                prediction = nn.LogSoftmax(prediction).max(2)[1]
                
                loss_batch = loss(prediction, y)
                losses.append(loss_batch.data[0])
                
                optimizer.zero_grad()
                loss_batch.backward()
                optimizer.step()
  
            train_loss_epochs.append(np.mean(losses))
            losses = []
            for sample in test_dataloader:
                X = sample['image']
                X = Variable(X)
                y = sample['anns']
                
                y = Variable(y)
                
                prediction = network(X)
                loss_batch = loss(prediction, y)
                losses.append(loss_batch.data[0])
                
            test_loss_epochs.append(np.mean(losses))
            
            is_best = test_loss_epochs[-1] < best_test_score
            best_test_score = min(test_loss_epochs[-1], best_test_score)
            save_checkpoint({
                            'epoch': epoch + 1,
                            'state_dict': network.state_dict(),
                            'best_test_score': best_test_score,
                            'optimizer' : optimizer.state_dict(),
                            }, is_best)
                
            
            sys.stdout.write('\rEpoch {0}... (Train/Test) MSE: {1:.3f}/{2:.3f}'.format(
                        epoch, train_loss_epochs[-1], test_loss_epochs[-1]))
    except KeyboardInterrupt:
        pass
    plt.figure(figsize=(12, 5))
    plt.plot(train_loss_epochs[1:], label='Train')
    plt.plot(test_loss_epochs[1:], label='Test')
    plt.xlabel('Epochs', fontsize=16)
    plt.ylabel('Loss', fontsize=16)
    plt.legend(loc=0, fontsize=16)
    plt.grid('on')
    plt.savefig('lstm_training.png')
  #  plt.show()

