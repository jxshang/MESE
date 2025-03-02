import torch
import torch.nn as nn
import torch.nn.functional as F

class habit_aware_comments_learning(nn.Module):
    def __init__(self,input_dim,hidden_dim):  
        super(habit_aware_comments_learning,self).__init__()
        self.query = nn.Linear(input_dim,hidden_dim)
        self.key = nn.Linear(input_dim,hidden_dim)
        self.value = nn.Linear(input_dim,hidden_dim)
    def forward(self,comments,sentences):  
        q = self.query(comments)
        k = self.key(sentences)
        v = self.value(sentences)
        attention_weights = F.softmax(torch.bmm(q,k.transpose(1,2))/(k.size(-1)**0.5),dim =-1) 
        output = torch.bmm(attention_weights,v)
        output = output.squeeze(1)
        return output


class comment_envolution(nn.Module):
    def __init__(self,input_dim,hidden_dim):
        super(comment_envolution,self).__init__()
        self.lstm = nn.LSTM(input_size = input_dim,hidden_size=hidden_dim,batch_first=True,bidirectional=False)
    def forward(self,input_data):
        output,last_time=self.lstm(input_data)
        return last_time[0].permute(1,0,2)
       

class sentence_bilstm(nn.Module):
    def __init__(self,input_dim,hidden_dim):
        super(sentence_bilstm,self).__init__()
        self.bilstm = nn.LSTM(input_size=input_dim,hidden_size=hidden_dim,batch_first=True,bidirectional=True)
    def forward(self,sentence):
        output,last_time= self.bilstm(sentence)  
        last_time = last_time[0].permute(1,0,2)
        return last_time
        
class emotion_envolution(nn.Module):
    def __init__(self,input_dim,hidden_dim):   
        super(emotion_envolution,self,).__init__()    
        self.emo_envolution = nn.LSTM(input_size=input_dim,hidden_size=hidden_dim,batch_first = True,bidirectional=False)
        self.fc = nn.Linear(hidden_dim,hidden_dim)
    def forward(self,emotion_input):
        emo_every,emo_last = self.emo_envolution(emotion_input)
        return emo_every,emo_last[0].permute(1,0,2)
    
class senmantic_emotion_fusion(nn.Module):   
    def __init__(self,semantic_dim,args,emotion_dim):
        super(senmantic_emotion_fusion,self).__init__()  
        self.fu = nn.Linear(semantic_dim+emotion_dim,args.h5)
        self.fc = nn.Linear(emotion_dim,args.h5) 
        self.fs = nn.Linear(semantic_dim,args.h5) 
        self.lstm = nn.LSTM(input_size=args.h5,hidden_size=args.h5,batch_first=True,bidirectional=False)

    def forward(self,semantic,emotion):
        combine_vector = torch.cat((semantic,emotion),dim=-1)
        batch_size,comment_length,_ = combine_vector.size()
        u = torch.sigmoid(self.fu(combine_vector.view(batch_size*comment_length,-1))) 
        c = torch.tanh(self.fc(emotion.reshape(batch_size*comment_length,-1))) 
        semantic = torch.tanh(self.fs(semantic.reshape(batch_size*comment_length,-1)))   
        semantic = semantic.reshape(batch_size,comment_length,-1)
        c = c.reshape(batch_size,comment_length,-1)
        u_expand = u.reshape(batch_size,comment_length,-1)
        n = u_expand*semantic+(1-u_expand)*c  #
        output,last_time= self.lstm(n)
        last_time = last_time[0].permute(1,0,2)
        return last_time
        
class My_model(nn.Module):
    def __init__(self,args):
        super(My_model,self).__init__()
        self.habit_attention = habit_aware_comments_learning(args.semantic_input_dim,args.h2)
        self.comment_envolution = comment_envolution(args.h2,args.h4)
        self.sentence_bilstm = sentence_bilstm(args.semantic_input_dim,args.h1)
        self.emo_envolution = emotion_envolution(args.emotion_input_dim,args.h3)
        self.semantic_emotion = senmantic_emotion_fusion(args.h2,args,args.emotion_input_dim) 
        self.fusion = nn.Linear(args.h1*2+args.h4+args.h5+args.h3,args.output_dim)
    
    def forward(self,comments,sentence,emotion):
        habit_comment_output = self.habit_attention(comments,sentence)
        comment_envolution = self.comment_envolution(habit_comment_output)
        sentence_embedding = self.sentence_bilstm(sentence)
        emotion_every,emo_final = self.emo_envolution(emotion)
        semantic_emo = self.semantic_emotion(habit_comment_output,emotion)
        sentence_embedding = sentence_embedding.reshape(sentence_embedding.shape[0],1,-1)
        concate_tensor = torch.cat((sentence_embedding,semantic_emo,comment_envolution,emo_final),dim=2)  
        reshaper_tensor = concate_tensor.view(concate_tensor.size(0),-1)
        output = self.fusion(reshaper_tensor)
        classfy_result = torch.relu(output)
        classfy_result = torch.sigmoid(classfy_result)
        
        return classfy_result  
    





