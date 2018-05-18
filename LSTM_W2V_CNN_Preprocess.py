
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


# In[3]:


gpu_device = 3
#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
#torch.cuda.set_device(gpu_device)
#.cuda(gpu_device)


# In[4]:


DEF_SEND = '<SEND>'
DEF_START = '<START>'


# In[5]:


TRAIN_DATSET_FILE = 'traindataset_cnn.tar.gz'
TEST_DATSET_FILE = 'testdataset_cnn.tar.gz'


# In[6]:


dataDir='/home/p.zaydel/ProjectNeuralNets/coco_dataset/'
imagesDirTrain = '{}train2017/train2017'.format(dataDir)
imagesDirVal = '{}val2017/val2017'.format(dataDir)

annTrainFile = '{}/annotations_trainval2017/annotations/captions_train2017.json'.format(dataDir)
annValFile = '{}/annotations_trainval2017/annotations/captions_val2017.json'.format(dataDir)


# In[7]:


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


# In[8]:



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
    Texts = list(dataset.anns.values())
    model = Word2Vec(Texts, size = embed_size, workers = 7, min_count = 0)
    return model

def generate_vocab_dicts(dataset): 
    Texts = list(dataset.anns.values())
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


# In[9]:


def save_prepared_dataset(dataset, filename, cnn_model = models.alexnet(pretrained=True).features):
    
    cnn_model = cnn_model.cuda(gpu_device)
    
    print("preparing images...")
    for idx in tqdm(range(len(dataset))):
        sample = dataset[idx]
        var = Variable(sample['image'].unsqueeze(0)).cuda(gpu_device)
        dataset.images_cnn[idx] =  cnn_model(var).data.view(-1).cpu()
        
    print("preparing annotations...")
    dataset.text_transform = split_text2words
    dataset.preload_anotations()
    dataset.text_transform = None
    
    torch.save(dataset, filename)
    print("Dataset saved in {}".format(filename))


# In[10]:


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


# In[11]:


if os.path.exists(TRAIN_DATSET_FILE):
    print("loading train dataset...")
    trainDataset = torch.load(TRAIN_DATSET_FILE)
    print('train dataset loaded!')
else:
    trainDataset = MSCOCODataset(annTrainFile,imagesDirTrain, transform = transform_to224, mode='pic2rand')
    save_prepared_dataset(trainDataset, TRAIN_DATSET_FILE, cnn_model)
    
if os.path.exists(TEST_DATSET_FILE):
    print("loading test dataset...")
    testDataset = torch.load(TEST_DATSET_FILE)
    print('test dataset loaded!')
else:
    testDataset = MSCOCODataset(annValFile,imagesDirVal, transform = transform_to224, mode='pic2rand')
    save_prepared_dataset(testDataset, TEST_DATSET_FILE, cnn_model)


# In[12]:


import os
print("Creating dictionary......")
if os.path.exists('dictionaries_2.tar.gz'):
    print("loading dictionary")
    dic_state = torch.load('dictionaries.tar.gz')
    words2ids = dic_state['words2ids']
    ids2words = dic_state['ids2words']
    print("dictionary loaded")
else:
    words2ids, ids2words  = generate_vocab_dicts(trainDataset)
    print("saving dictionary")
    torch.save({'words2ids': words2ids, 'ids2words': ids2words }, 'dictionaries.tar.gz')


# In[13]:


# MY WORD EMBEDDINGS
import os

WORD_EMBED_FILE = 'word_embeding_6.tar.gz'
if os.path.exists(WORD_EMBED_FILE):
    print("loading words embedding")
    word_embeding = torch.load(WORD_EMBED_FILE)
    print("words embedding loaded")
else:
    print("creating words embedding......")
    word_embeding = train_word_to_vec_gensim(trainDataset, embed_size = 300)
    print("saving words embedding")
    torch.save(word_embeding, WORD_EMBED_FILE)


# In[14]:



class LSTM_W2V_Net_Cnn_Preload(nn.Module):

    def __init__(self,  image_size, image_features_size, word_embedding, words2ids, ids2words,
                 lstm_hidden_size = 2000,
                 word_embedding_size = 500, 
                 cnn = models.alexnet(pretrained=True).features,
                 start_symbol = DEF_START,
                 end_symbol = DEF_SEND
              #   cnn_comp_features = lambda cnn, x: cnn.features(x),
              #   max_sentence_len = 20,
              #   sentence_end_embed = None,
             #  sentence_end_symbol = '.'
                  ):
        """Init NN
            image_size - size of input image.
            lstm_hidden_size - size of cnn features output
            image_features_size - size of image features vector
            word_embedding - pretrained word embedding model
            words2ids - dictionary word -> id
            ids2words - dictionary id -> word
            cnn - pretrained cnn net (alexnet, vgg and other)
            start_symbol - symbol starting sequence
            end_symbol - symbol ending sequence
        """
        
        super(LSTM_W2V_Net_Cnn_Preload, self).__init__()
        self.image_size = image_size
        self.image_features_size = image_features_size
        #self.cnn = cnn
     #   self.cnn_comp_features = cnn_comp_features
        
        self.vocab_size = len(words2ids)
        
        self.word_embedding_size = word_embedding_size
        self.word_embedding = word_embedding
        
        self.words2ids = words2ids
        self.ids2words = ids2words
        
        self.start_symbol = start_symbol
        self.start_symbol_embed = torch.from_numpy(self.word_embedding[self.start_symbol])
        
        self.end_symbol = end_symbol
        self.end_symbol_embed = torch.from_numpy(self.word_embedding[self.end_symbol])
        
#         self.sentence_end_symbol = sentence_end_symbol
#         self.sentence_end_symbol_id = self.words2ids[self.sentence_end_symbol]
        
#         if sentence_end_embed is not None:
#             self.sentence_end_embed = sentence_end_embed
#         else:
#             self.sentence_end_embed = word_embeding['.']
        
        #self.max_sentence_len = max_sentence_len
        
        
        self.lstm_hidden_size = lstm_hidden_size
        
        self.fc1 = nn.Sequential( nn.BatchNorm1d(self.image_features_size),
                                  nn.Linear(self.image_features_size, int(self.image_features_size/2)),
                                  nn.Dropout(0.001), 
                                  nn.ReLU(),
                                  nn.Linear(int(self.image_features_size/2), int(self.image_features_size/4) ),
                                  nn.Dropout(0.001),
                                  nn.ReLU(),
                                  nn.Linear(int(self.image_features_size/4), self.lstm_hidden_size),
                                  nn.BatchNorm1d(self.lstm_hidden_size)
                                )
        
#         self.fc1 = nn.Sequential( nn.BatchNorm1d(self.image_features_size),
#                                   nn.Linear(self.image_features_size, self.lstm_hidden_size),
# #                                   nn.Dropout(0.001), 
# #                                   nn.ReLU(),
# #                                   nn.Linear(int(self.image_features_size/4), self.lstm_hidden_size),
#                                   nn.BatchNorm1d(self.lstm_hidden_size)
#                                 ).cuda(gpu_device)
        
        self.fc2 = nn.Sequential(nn.Linear(self.lstm_hidden_size, self.vocab_size),
                                  nn.LogSoftmax()
                                )
        
                               
        self.lstm_cell = nn.LSTMCell(self.lstm_hidden_size + self.word_embedding_size, 
                                     self.lstm_hidden_size)
        
#         self.lstm = nn.LSTM(self.lstm_hidden_size , word_embedding_size)
    
        
    
#     def freeze_cnn(self):
#         for param in self.cnn.parameters():
#             param.requires_grad = False
    
#     def unfreeze_cnn(self):
#         for param in self.cnn.parameters():
#             param.requires_grad = True

    def set_mode(self, mode):
        if mode == 'train':
            for layer in self.fc1:
                layer.training = True
                
            for layer in self.fc2:
                layer.training = True
        elif mode == 'test':
            for layer in self.fc1:
                layer.training = False
                
            for layer in self.fc2:
                layer.training = False
            
    def ids_to_embed(self, word_ids):
        result = []
        
        for i in range(word_ids.shape[0]):
            w = self.ids2words[word_ids[i].data[0]]
            
            emb = torch.from_numpy(self.word_embedding[w]).float()
            result.append(emb)
            
        return torch.stack(result)
        
            
            
    def forward(self, X, max_sentence_len):
        batch_size = X.shape[0]
        
        
        #X = self.cnn(X)
        X = X.view(batch_size, self.image_features_size)
        X = self.fc1(X)
        
        
        # prevWord = START_SYMBOL
        prevWord = Variable(self.start_symbol_embed.repeat(batch_size, 1), requires_grad=True)

#         print('X', X.shape)
#        print('pW', prevWord.shape)
        lstm_input = torch.cat([X, prevWord], dim = 1)
        
        result = []
        
        h_t = Variable(torch.zeros(batch_size, self.lstm_hidden_size), requires_grad=False)
        c_t = Variable(torch.zeros(batch_size, self.lstm_hidden_size), requires_grad=False)
        
#         print(lstm_input.shape)
#         print(h_t.shape)
#         print(c_t.shape)
        
        for idx in range(max_sentence_len):
            h_t, c_t = self.lstm_cell.forward(lstm_input, (h_t, c_t))
            probs = self.fc2.forward(h_t)
            
            top_word_ids = probs.max(1)[1]
            embeds = self.ids_to_embed(top_word_ids)
            embeds = Variable(embeds)
            
            
#             return embeds
            
#             print(prevWord)
            
#             print(X.shape)
#             print(embeds.shape)
            
#             print(embeds)
            
            lstm_input = torch.cat([X, embeds], dim = 1)
#             print(probs.shape)
            result.append(probs)
            
        #return result
        result = torch.stack(result, dim = 1)
        return result
        


# In[15]:


image_size = 224
image_features_size = get_alexnet_features_dim(image_size)

lstmnet = LSTM_W2V_Net_Cnn_Preload(image_size, image_features_size, 
                       word_embeding, words2ids, ids2words, 
                       word_embedding_size = word_embeding.layer1_size)


# In[16]:


# print("START NN TEST")

# trainDataLoader_2 = DataLoader(trainDataset, batch_size = 2, shuffle=True)
# for sample in trainDataLoader_2:
#     break

# loss = nn.NLLLoss()
# optimizer = torch.optim.Adam(lstmnet.parameters(), lr=0.001)


# optimizer.zero_grad()

# X = sample['image']

# X = Variable(X)

# ann_ids = sample['anns']

# batch_size = X.shape[0]
# max_len = sample['ann_len'].max()

# y = load_anns(trainDataset, ann_ids, max_len, prepare=lambda w: words2ids[w])
# y = Variable(y.long())

# pred = lstmnet.forward(X, max_len)

# pred = pred.cpu()
# y = y.cpu()

# pred_b = pred.view(pred.shape[0]*pred.shape[1], pred.shape[2])

# loss_batch = loss(pred_b, y.view(-1))

# loss_batch.backward()
# optimizer.step()

# print(loss_batch)
# print("TEST SUCCESS")


# In[17]:


# sample = trainDataset[0]
# X = sample['image']
# X = Variable(X).unsqueeze(0)
# lstmnet.set_mode('test')
# pred = lstmnet.forward(X, max_len)
# lstmnet.set_mode('train')


# In[18]:


# loss = nn.CrossEntropyLoss()
# input =Variable(torch.randn(2, 3, 5), requires_grad=True)
# target = Variable(torch.LongTensor(2,3).random_(5))

# target_onehot = torch.zeros(target.shape[0], target.shape[1],  5)
# for i in range(target.shape[0]):
#     target_onehot[i].scatter_(1, target.data[i].unsqueeze(1) , 1)

# target_onehot = Variable(target_onehot)
# target_onehot = target_onehot.view(target_onehot.shape[0]*target_onehot.shape[1], target_onehot.shape[2])
# input = input.view(input.shape[0]*input.shape[1], input.shape[2])

# loss(input, target.view(-1))


# In[19]:


def save_checkpoint(state, is_best, filename='checkpoint_1.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'best_'+filename)
        
def open_checkpoint(is_best = False, filename='checkpoint_1.pth.tar'):
    if is_best:
        filename = 'best_'+filename
        
    checkpoint = torch.load(filename)
    return checkpoint
#     best_prec1 = checkpoint['best_prec1']
#     model.load_state_dict(checkpoint['state_dict'])
#     optimizer.load_state_dict(checkpoint['optimizer'])


# In[20]:


TEST_SAMPLE_ID =3455
def test_nn_on_image(network, sample_idx = TEST_SAMPLE_ID):
    network.set_mode('test')
    sample = trainDataset[sample_idx]
    ann_id = sample['anns']
    ann = trainDataset.get_ann(ann_id)
    im_id = sample['imid']
    X = Variable(sample['image']).unsqueeze(0)
    max_len = sample['ann_len']
    
    pred = lstmnet.forward(X, max_len)
    wids = pred[0].max(1)[1]
    
    result = []
    for i in range(wids.shape[0]):
        wid = wids[i].data[0]
        word = ids2words[wid]
        result.append(word)
        
    network.set_mode('train')
    
    return {'res': result, 'ann_id': ann_id, 'imid': im_id, 'ann': ann, 'max_len': max_len}
    
    


# In[21]:



TRAIN_LOG_FILE = "train_log_1.txt"
TRAIN_PLT_FILE = 'train_plt.png'

def train(network, train_dataloader, test_dataloader,
          epochs,  loss = nn.NLLLoss(), optim=torch.optim.Adam ):
    
    print("TRAIN STARTED!")
    log_file = open(TRAIN_LOG_FILE,'w') 
#     if unfreeze_cnn_epoch is None:
#         unfreeze_cnn_epoch = int(0.75 * epochs)
    
    train_loss_epochs = []
    test_loss_epochs = []
    optimizer = optim(network.parameters(), lr=0.001)
    best_test_score = 10**6
    
    sheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
#     network.freeze_cnn()
    
    try:
        for epoch in range(epochs):
            sheduler.step()

            losses = []
            accuracies = []
            for sample in tqdm(train_dataloader):

                #torch.cuda.empty_cache()
                optimizer.zero_grad()

                X = sample['image']

                X = Variable(X)

                ann_ids = sample['anns']

                batch_size = X.shape[0]
                max_len = sample['ann_len'].max()

                y = load_anns(trainDataset, ann_ids, max_len, prepare=lambda w: words2ids[w])
                y = Variable(y.long())

                pred = lstmnet.forward(X, max_len)

                pred = pred.cpu()
                y = y.cpu()

                pred_b = pred.view(pred.shape[0]*pred.shape[1], pred.shape[2])

                loss_batch = loss(pred_b, y.view(-1))
                losses.append(loss_batch)

                loss_batch.backward()
                optimizer.step()
                optimizer.zero_grad()
                del X, y, pred_b, pred, loss_batch
  
            train_loss_epochs.append(np.mean(losses))
            losses = []
        
            for sample in test_dataloader:
                #torch.cuda.empty_cache()
                X = sample['image']
                X = Variable(X)
                ann_ids = sample['anns']

                batch_size = X.shape[0]
                max_len = sample['ann_len'].max()

                y = load_anns(trainDataset, ann_ids, max_len, prepare=lambda w: words2ids[w])
                y = Variable(y.long())

                pred = lstmnet.forward(X, max_len)

                pred = pred.cpu()
                y = y.cpu()

                pred_b = pred.view(pred.shape[0]*pred.shape[1], pred.shape[2]).cpu()

                loss_batch = loss(pred_b.cpu(), y.view(-1).cpu())
                losses.append(loss_batch)
                del X, y, pred_b, pred, loss_batch
                
                
            test_loss_epochs.append(np.mean(losses))
            
            image_test = test_nn_on_image(network)
            
            log_file.write("Epoch:{}".format(epoch + 1))
            log_file.write("Mean Test Loss:{}".format(np.mean(losses)))
            log_file.write("Test on image:\n {}".format(image_test))
            
            
            is_best = test_loss_epochs[-1] < best_test_score
            best_test_score = min(test_loss_epochs[-1], best_test_score)
            save_checkpoint({
                            'epoch': epoch + 1,
                            'state_dict': network.state_dict(),
                            'best_test_score': best_test_score,
                            'optimizer' : optimizer.state_dict(),
                            }, is_best)
                
            
            sys.stdout.write('\rEpoch {0}... (Train/Test) Loss: {1:.3f}/{2:.3f}'.format(
                        epoch, train_loss_epochs[-1], test_loss_epochs[-1]))
    except KeyboardInterrupt:
        close(log_file)
        pass
    plt.figure(figsize=(12, 5))
    plt.plot(train_loss_epochs[1:], label='Train')
    plt.plot(test_loss_epochs[1:], label='Test')
    plt.xlabel('Epochs', fontsize=16)
    plt.ylabel('Loss', fontsize=16)
    plt.legend(loc=0, fontsize=16)
    plt.grid('on')
    plt.savefig(TRAIN_PLT_FILE)
    
    close(log_file)
    
    print("TRAIN ENDED!")


# In[23]:


trainDataLoader = DataLoader(trainDataset, batch_size = 64, shuffle=True)
testDataLoader = DataLoader(testDataset, batch_size = 64, shuffle=True)


# In[24]:


train(lstmnet, trainDataLoader, testDataLoader, 20, loss = nn.NLLLoss() )

