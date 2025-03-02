import pickle
def cat_all(name):
    #embedding
    with open(f"../Embedding/{name}_embedding.pickle",'rb') as f:
        text_feature = pickle.load(f)
    ##emotion
    with open(f"../Emotion/{name}_emo_dic.pickle",'rb') as f:
        emotion = pickle.load(f)
    print(len(text_feature),len(emotion))
    ##data
    with open(f"../dataset/before_process/{name}/All_clean_{name}.pickle",'rb') as f:
        newsdic  = pickle.load(f)
    ###cat##### 
    for id,newsinfo in newsdic.items():
        newsinfo["comments_feature"] = text_feature[id]["comments_feature"]
        newsinfo["sentence_feature"] =text_feature[id]["sentence_feature"]
        newsinfo["comments_emotion"] = emotion[id]["comments_emotion"]
    print([list(newsinfo.keys()) for id,newsinfo in list(newsdic.items())[:1]])


    text_feature = None
    emotion = None
    ##save
    with open(f"../dataset/after_process/{name}_dic.pickle",'wb') as f:
        pickle.dump(newsdic,f)
    with open(f"../dataset/after_process/{name}_dic.pickle",'rb') as f:
        news = pickle.load(f)
        print(len(news))
   
   
if __name__ =="__main__":
    dataset = ["RumourEval-19","Weibo-comp"]
    dataset = ["Weibo-comp"]
    for name in dataset:
        cat_all(name)
    print("All data has been processed successfully!Now,you can start run model!")