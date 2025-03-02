import pickle
import torch
from transformers import BertTokenizer,BertModel
import re
import numpy

def get_embedding(name):
    path = f"../dataset/before_process/{name}/All_clean_{name}.pickle"    
    with open(path,"rb") as f:
        newsdic = pickle.load(f)
    if(name=="RumourEval-19"):
        ##Local loading
        # tokenizer  = BertTokenizer.from_pretrained(r"C:\Users\Aurora\Desktop\2210.01\model\down\en")
        # model = BertModel.from_pretrained(r"C:\Users\Aurora\Desktop\2210.01\model\down\en")
        #Online Loading
        tokenizer  = BertTokenizer.from_pretrained("bert-base-uncase")
        model = BertModel.from_pretrained("bert-base-uncase")
        
    elif(name=="Weibo-comp"):
        #Local loading
        # tokenizer  = BertTokenizer.from_pretrained(r"C:\Users\Aurora\Desktop\2210.01\model\down")
        # model = BertModel.from_pretrained(r"C:\Users\Aurora\Desktop\2210.01\model\down")
        #Online Loading
        tokenizer  = BertTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
        model = BertModel.from_pretrained("hfl/chinese-roberta-wwm-ext")
    print("tokenzier and model have benn load successfully")
    text_feature = {}
    from tqdm import tqdm
    count = 0
    with torch.no_grad():
        for newsid,newsinfo in tqdm(newsdic.items(),total = len(newsdic)):
           
            before_clean_news_sentence_list = re.split('。|！|？',newsdic[newsid]['news_content'])
            news_sentence_list = [s.strip() for s in before_clean_news_sentence_list if s.strip()!='']
            comments_list = [comment.strip() for comment in newsdic[newsid]["comments"] if comment.strip()!='']
            
            if(len(comments_list)>200):
                comments_list= newsinfo["comments"][:200]
            news_feature = []
            comments_feature =[]
            sentence_feature = []

            for sentence in news_sentence_list:
                tok = tokenizer.encode(sentence)
                if(len(tok)>=512):
                    sentence_input_id = torch.tensor(tok[0:512]).unsqueeze(0)  
                else:
                    sentence_input_id = torch.tensor(tok).unsqueeze(0)
                sentence_output = model(sentence_input_id)
                sentence_feature .extend(numpy.array(sentence_output[1].detach().numpy()).tolist())
            print("news feature shape:",numpy.array(sentence_feature ).shape)
            text_feature[newsid] = {}
            text_feature[newsid]['sentence_feature'] = sentence_feature 

            for comments in comments_list:
                tok = tokenizer.encode(comments)
                if(len(tok)>=512):
                    comments_input_id = torch.tensor(tok[0:512]).unsqueeze(0)
                else:
                    comments_input_id = torch.tensor(tok).unsqueeze(0)
                comment_output = model(comments_input_id)    
                comments_feature.extend(numpy.array(comment_output[1].detach().numpy()).tolist())
            print("comments feature shape:",numpy.array(comments_feature).shape) 
            text_feature[newsid]['comments_feature'] = comments_feature
            count = count+1
                
        print("text embedding is done") 
        print("len of newsdic:",len(text_feature))
        
        for key in list(text_feature.keys())[:1]:
            print(key,text_feature[key])
            print("comments_feature:",text_feature[key]["comments_feature"])
        
        print("Now,saving embedding")
        save_path = f"../Embedding/{name}_embedding.pickle"
            
        with open(save_path,'wb') as f:
            pickle.dump(text_feature,f)
        with open(save_path, 'rb') as f:
            text_feature = pickle.load(f)
        print("All embedding work have benn finished!")  
           
if __name__ =="__main__":
    dataset = ["RumourEval-19","Weibo-comp"]
    get_embedding("Weibo-comp")

    

