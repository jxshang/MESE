from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset,DataLoader
import torch.nn.functional as F
import torch
import pickle
from settings import args
import numpy


if(torch.cuda.is_available()):
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
class MyDataset(Dataset):
    def __init__(self,comments,sentences,emotions,label):
        self.comments = comments
        self.sentences = sentences
        self.emotions = emotions
        self.labels = label
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        sample_data = (self.comments[idx],self.sentences[idx],self.emotions[idx])
        label = self.labels[idx]
        return sample_data,label


def con_dataloader(name):
    ##read embedding and emotion
    with open(f"dataset/after_process/{name}_dic.pickle",'rb') as f:
        newsdic = pickle.load(f)
    ##get news_id of val、train、test set 
    with open(f"dataset/before_process/{name}/train_test_val/train.pickle",'rb') as f:
        train_dic = pickle.load(f)
        train_list = list(train_dic.keys())
        train_dic = None
    with open(f"dataset/before_process/{name}/train_test_val/test.pickle",'rb') as f:
        test_dic = pickle.load(f)
        test_list=list(test_dic.keys())
        test_dic = None
    with open(f"dataset/before_process/{name}/train_test_val/val.pickle",'rb') as f:
        val_dic = pickle.load(f)
        val_list = list(val_dic.keys())
        val_dic = None   
        
    data = ['train','test','valid']
    for dty in data:
        if(dty =='train'):
            data_id = train_list
        elif(dty=="test"):
            data_id=test_list
        elif(dty=="valid"):
            data_id = val_list
        
        data_labels =  torch.tensor(numpy.array([newsdic.get(id)["lable"] for id in data_id]))
        data_comments = [torch.tensor(newsdic.get(id)['comments_feature'][:args.Max_Comment_Number]).float() for id in data_id]
        data_sentences = [torch.tensor(newsdic.get(id)['sentence_feature'][:args.MAX_sentences]).float() for id in data_id]
        data_emotions =[torch.tensor(newsdic.get(id)['comments_emotion'][:args.Max_Comment_Number]).float() for id in data_id]
        
        padded_comments = pad_sequence(data_comments,batch_first=True,padding_value=0)
        padded_sentences = pad_sequence(data_sentences,batch_first=True,padding_value=0)
        padded_emotion = pad_sequence(data_emotions,batch_first=True,padding_value=0)
        
        print(f"The feature shape of  {dty} set : ")
        print(f"padded_comments:{padded_comments.shape},padded_sentences:{padded_sentences.shape}")
        print(f"padded_emotion:{padded_emotion.shape},data_labels:{data_labels.shape}")
        
        if dty == 'train':
           train_dataset = MyDataset(padded_comments,padded_sentences,padded_emotion,data_labels)
        elif dty == 'test':
           test_dataset = MyDataset(padded_comments,padded_sentences,padded_emotion,data_labels)
        elif dty =='valid':
           valid_dataset = MyDataset(padded_comments,padded_sentences,padded_emotion,data_labels)

    train_loader = DataLoader(train_dataset,batch_size=args.batch_size,shuffle=True)
    test_loader = DataLoader(test_dataset,batch_size=args.batch_size,shuffle=False)
    valid_loader = DataLoader(valid_dataset,batch_size=args.batch_size,shuffle=False)
    return train_loader,test_loader,valid_loader

train_loader, test_loader,val_loader = con_dataloader(args.dataset)
print("data loaded successfully")