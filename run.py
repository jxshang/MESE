import torch 
from model import My_model  
import torch.nn.functional as F
from settings import args
from data_input import train_loader,test_loader,val_loader
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
import numpy
if(torch.cuda.is_available()):
    device = "cuda:0"
else :
    device = "cpu"


def train(model,train_loader,args,valid_loader,test_loader):  
    optimizer = torch.optim.Adam(model.parameters(),lr = args.learning_rate,weight_decay=args.weight_decay)
    criterion = torch.nn.CrossEntropyLoss()
    best_valid_loss = float('inf')  
    
    for epoch in range(args.num_epochs):
        model.train()
        epoch_loss = 0
        for batch_data in train_loader:
            (comments_embedding,sentence_embedding,emotion_feature),labels = batch_data
            comments_embedding = comments_embedding.to(device)
            sentence_embedding = sentence_embedding.to(device)
            emotion_feature = emotion_feature.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(comments_embedding,sentence_embedding,emotion_feature)
            loss = criterion(outputs,labels)
            epoch_loss = epoch_loss+loss.item()
            loss.backward()
            optimizer.step()
        print("*"*80,f"Epoch{epoch+1}/{args.num_epochs},Loss:{epoch_loss}")

        print("Testing on test set ")
        if args.dataset =="RumourEval-19":
            f1_test,rmse_test= test(model,test_loader)
            f1,rmse = test(model,val_loader)
        else:
            a_test,p_test,r_test,f_test =  test(model,test_loader)
            a,p,r,f = test(model,val_loader)

        valid_loss = calculate_valid_loss(model,valid_loader,criterion)
        print(f'Epoch {epoch+1}/{args.num_epochs},Loss:{epoch_loss},Valid Loss:{valid_loss}')

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            improve_count = 0
            if args.dataset =="RumourEval-19":
                best_result = [f1,rmse]
            else:
                best_result = [a_test,p_test,r_test,f_test]
        else:
            improve_count = improve_count +1
            if improve_count >= args.patience:
                print("Early stopping...")
                print("saving model parameters...")
                import os
                if not os.path.exists(f'param/{args.dataset}'):
                    os.makedirs(f'param/{args.dataset}')
                torch.save(model.state_dict(),f'param/{args.dataset}/{args.time}_{args.dataset}_save_model.pth')
                print("saving successfully!")
                return best_result
        
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
def test(model,test_loader):
    model.eval()
    y_true=[]
    y_pre = []
    with torch.no_grad():
        for inputs,labels in test_loader:
            (comments_embedding,sentence_embedding,emotion_feature) = inputs
            comments_embedding = comments_embedding.to(device)
            sentence_embedding = sentence_embedding.to(device)
            emotion_feature = emotion_feature.to(device)
            labels = labels.to(device)
            outputs = model(comments_embedding,sentence_embedding,emotion_feature)  
            _, predicts = torch.max(outputs,1)  
            y_true.extend(labels.cpu().numpy())
            y_pre.extend(predicts.cpu().numpy())
       
        if args.dataset=='RumourEval-19':
            f1 = f1_score(y_true,y_pre,average="macro")
            rmse = torch.sqrt(F.mse_loss(torch.tensor(y_true).float(),torch.tensor(y_pre).float()))
            print(f"f1_macro:{f1},RMSE:{rmse.item()}")
            return f1,rmse
        else:
            accuracy = accuracy_score(y_true,y_pre)
            precision = precision_score(y_true,y_pre)
            recall = recall_score(y_true,y_pre)
            f1 = f1_score(y_true,y_pre)
            print(accuracy,precision,recall,f1)
            return accuracy,precision,recall,f1
             
            
            
def calculate_valid_loss(model,valid_loader,criterion): 
    model.eval()
    valid_loss = 0.0
    with torch.no_grad():
        for data in valid_loader:
            (comments_embedding,sentence_embedding,emotion_feature),labels = data
            comments_embedding = comments_embedding.to(device)
            sentence_embedding = sentence_embedding.to(device)
            emotion_feature = emotion_feature.to(device)
            labels = labels.to(device)
            outputs = model(comments_embedding,sentence_embedding,emotion_feature)
            loss = criterion(outputs,labels)
            valid_loss +=loss.item()
        return valid_loss   

import os
def save_result_weibocomp(a,b,c,d,flag="every"):
    print("Begin saving data...")
    if not os.path.exists(f'result'):
        os.makedirs(f"result")
        
    with open(f'result/{args.dataset}_result.txt','a+') as f:
        f.write(f"-------{flag}-----------lr={args.learning_rate}------------\n")
        f.write(f'Acc:{a}')
        f.write(" "*10)
        f.write(f'Pre:{b}')
        f.write(" "*10)
        f.write(f'Recall:{c}')
        f.write(" "*10)
        f.write(f'F1:{d}\n')
        f.write("---------------------------------------------")
        f.close()


def save_result_r19(a,b,flag="every"):
    print("save result...")
    
    with open(f'result/{args.dataset}_result.txt','a+') as f:
        f.write(f"---------{flag}---------lr={args.learning_rate}------------\n")
        f.write(f'F1:{a}')
        f.write(" "*10)
        f.write(f'RMSE:{b}\n')
      
        f.write("---------------------------------------------")
        f.close()

if __name__ == '__main__':
    result_array = []
    if not os.path.exists('result'):
        os.makedirs('result')
    for i in range(5):
        model = My_model(args)
        model.to(device)
        if(args.dataset=="RumourEval-19"):
            print("Begin training...")
            f1,rmse=train(model,train_loader,args,val_loader,test_loader)
            save_result_r19(f1,rmse)
            result_array.append([f1,rmse])
            if(i==4):
                a,b=numpy.array(result_array).mean(axis=0)
                save_result_r19(a,b,"avg")
                
        else:
            print("Begin training...")
            a,b,c,d=train(model,train_loader,args,val_loader,test_loader)
            save_result_weibocomp(a,b,c,d)    
            result_array.append([a,b,c,d])
            if(i==4):
                a,b,c,d=numpy.array(result_array).mean(axis=0)
                save_result_weibocomp(a,b,c,d,"avg")
    
    
    