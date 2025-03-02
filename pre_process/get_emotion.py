import sys
sys.path.append("../../dependencies/twitter-emotion-recognition-master")

from emotion_predictor import EmotionPredictor
import pickle
from tqdm import tqdm
import re
import json
import requests
import os
import pandas

def extract_emo_english(name):
    emo_dic={}
    with open(f"../../dataset/before_process/{name}/All_clean_{name}.pickle",'rb') as f:
        newsdic = pickle.load(f)
    model = EmotionPredictor(classification='ekman',setting='mc') 
    
    for newsid,newsinfo in tqdm(newsdic.items(),desc="Processing"):
        comments_list = newsinfo["comments"]
        if(len(comments_list)>200):
            comments_list = comments_list[:200]  
        emo_dic[newsid]={}
        predictions = model.predict_classes(comments_list)  # Anger,Disgust, Fear,Joy,Sadness,Surprise
        print(predictions, '\n')
        probabilities = model.predict_probabilities(comments_list)
        print(probabilities, '\n')
        select_values = probabilities.iloc[:,1:].values
        print(select_values)
        print(select_values.shape)
        emo_dic[newsid]["comments_emotion"]=select_values
        
        ##get news_emotion
        news_list = [newsinfo["news_content"]]
        probabilities = model.predict_probabilities(news_list)
        select_values = probabilities.iloc[:,1:].values
        print(select_values)
        print(select_values.shape)
        emo_dic[newsid]["news_emotion"]=select_values
         
    print("finished")
    with open(f"../../Emotion/{name}_emo_dic.pickle",'wb') as f:
        pickle.dump(emo_dic,f)
    print("save successfully!")
  
API_KEY = "xxx"
SECRET_KEY = "xxx"

def get_access_token():
    """
    使用 AK，SK 生成鉴权签名（Access Token）
    :return: access_token，或是None(如果错误)
    """
    url = "https://aip.baidubce.com/oauth/2.0/token"
    params = {"grant_type": "client_credentials", "client_id": API_KEY, "client_secret": SECRET_KEY}
    return str(requests.post(url, params=params).json().get("access_token"))

  
def extract_emo_chinese(name):
    path = f"../dataset/before_process/{name}/All_clean_{name}.pickle"
    with open(path,'rb') as f:
       newsdic = pickle.load(f)

    ##The URL retrieval method is described in the ReadMe.md file in this folder.
    url = "xxx" + get_access_token()
    emotion_dic = {}
    count=0
    
    A_count = 0;
    Flag = False
    record_list = []
    len_record = 0;
    for newsid,newsinfo in tqdm(newsdic.items(),desc="Processing newsdic"):
        if(len(record_list)>len_record):  
            len_record = len(record_list)
            with open("record.txt",'a+') as f:
                f.write(f"{record_list[-1]}\n")
        
        A_count = A_count+1   ##Statistics on the amount of data processed
        save_list = []
        text_list = []
        comments_list = newsdic[newsid]["comments"]
        if(len(comments_list)>200):
            comments_list = comments_list[:200]
        news= newsdic[newsid]['news_content']
        text_list.append(news)
        text_list.extend(comments_list)
        print("the number of comment is",len(comments_list),"the length of text_list is ",len(text_list))
        count = 0;
        import time
        emotion_dic[newsid]={}
        for text in text_list:
            time.sleep(0.5)
            payload = json.dumps({"text":text})
            headers = {
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            }
            response = requests.request("POST", url, headers=headers, data=payload)
            save_list.append(response.text)
            
            if("error_code" in response.text):
                record_list.append(newsid)
            if("\"error_code\":18" in response.text):
                print("error!",newsid,text,response.text)
                print("############################error###########################")
                Flag = True
                break
            if(count==0):
                emotion_dic[newsid]["news_emotion"]=save_list.copy()
                count = count+1
                save_list.clear()
          
        emotion_dic[newsid]['comments_emotion'] = save_list
        #Segmented storage
        if(Flag==True):break
        #if(A_count%100==0 or A_count>(len(newsdic)-len(newsdic)%1000)):
            # if not os.path.exists(f"../Emotion/{name}"):
            #     os.makedirs(f"../Emotion/{name}")
            # path = f"../Emotion/{name}/raw_emotion_{count}.pickle"
            # with open(path,'wb') as f:   
            #     pickle.dump(emotion_dic, f)
     
        ##Store the entire dataset together
        if not os.path.exists(f"../Emotion/{name}"):
            os.makedirs(f"../Emotion/{name}")
        path = f"../Emotion/{name}/raw_emotion.pickle"
        with open(path,'wb') as f:   
            pickle.dump(emotion_dic, f)

def process_baidu_data(name):
    with open(f'../Emotion/{name}/raw_emotion.pickle','rb') as f:
        emotion_dic = pickle.load(f)

    pre_emotion_dic={}

    print("#################Read emotion items obtained from the API and perform preprocessing###################################")
    for newsid,newsinfo in emotion_dic.items():
        pre_emotion_dic[newsid]={}
        for type_c,info in newsinfo.items():
            pre_emotion_dic[newsid][type_c]={}
            print("The parsed text type is:",type)
            if(type_c=="comments_emotion"):
                comment_emo_list=[]
            for ite in info:  
                if ite=="":continue
                sub_dict = eval(ite)
                if("error_code" in list(sub_dict.keys())):  ##If the text cannot be parsed, assign the value: Neutral.
                    if type_c=="news_emotion":
                        temp_dic = {}
                        temp_dic["neutral"] = 0
                        pre_emotion_dic[newsid][type_c] =temp_dic
                    elif type_c=="comments_emotion":
                        temp_dic = {}
                        temp_dic["neutral"]=0
                        comment_emo_list.append(temp_dic)
                    continue
                temp_dic={}
                
                item_list = sub_dict["items"]
                for emotion_item in item_list:
                    label= emotion_item["label"]
                    prob = emotion_item["prob"]
                    temp_dic[label]=prob
                    if 'subitems' in emotion_item:
                        subitems = emotion_item['subitems']
                        for subitem in subitems: 
                            sub_pro = subitem['prob']
                            sub_label = subitem['label']
                            temp_dic[sub_label]=sub_pro
                if(type_c=="comments_emotion"):
                    comment_emo_list.append(temp_dic)
            
            if(type_c=="comments_emotion"):
                pre_emotion_dic[newsid][type_c]=comment_emo_list
            elif type_c=="news":
                pre_emotion_dic[newsid][type_c]=temp_dic   
  
    return pre_emotion_dic  

  
def extract_feature(pre_emotion_dic,name):
    print("Build emotion dictionary")
    emotion_dic = {}
    set1 = []
    set2= []
    for newsid,newsinfo in pre_emotion_dic.items():
        emotion_dic[newsid]={}
        a_dic = pre_emotion_dic[newsid]['news_emotion']
        b_dic = pre_emotion_dic[newsid]['comments_emotion']
        print(a_dic)
        set1.extend(list(a_dic.keys()))
        for i in b_dic:
            set2.extend(list(i.keys()))
        #comments emotion type
        key_orders=['like','happy','optimistic','neutral', 'sad', 'pessimistic', 'fearful', 'disgusting', 'angry']

        a_key=['like', 'sad', 'pessimistic', 'angry', 'fearful', 'disgusting', 'optimistic', 'neutral', 'happy']
        df_news = pandas.DataFrame(list(a_dic.items()),columns=['Key','Value'],dtype=object)
        df_comments=pandas.DataFrame(b_dic)
        
        for col in key_orders:
            if col not in df_comments.columns:
                df_comments[col] = 0
    

        df_news=df_news.set_index('Key').reindex(a_key)
        emo_news = df_news['Value'].fillna(0).values  
        emo_comments = df_comments[key_orders].fillna(0).values
        
        print(emo_comments,emo_comments.shape)
        emotion_dic[newsid]['news_emotion'] = emo_news
        emotion_dic[newsid]["comments_emotion"] = emo_comments
        
    print(set(set1),set(set2))  
    
    with open(f'../Emotion/{name}/{name}_emo_dic.pickle','wb') as f:
        pickle.dump(emotion_dic,f)
    with open(f'../Emotion/{name}/{name}_emo_dic.pickle',"rb") as f:
        dic = pickle.load(f)
        
  
if __name__ == "__main__":
    dataset = ["RumourEval-19","Weibo-comp"]
    dataset = ['Weibo-comp']
    for name in dataset:
        if name=="RumourEval-19":
            extract_emo_english(name)
        elif name =="Weibo-comp":
            print("Call the API to get raw emotion data")
            extract_emo_chinese(name)
            print("Process the returned data to obtain a dictionary")
            pre_emotion_dic=process_baidu_data(name)       
            print("Process the dictionary to obtain feature vectors")
            extract_feature(pre_emotion_dic,name)     
    
    