# The configuration below is from the paper.
import argparse
import datetime

parser = argparse.ArgumentParser(description="the model's Parameters")
current_time = datetime.datetime.now()
formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
print("Current time:", formatted_time)
time = formatted_time
parser.add_argument("--time",type=str,default=time)

#########################################r19####################################

# parser.add_argument("--semantic_input_dim",type=int,default=768)
# parser.add_argument("--hidden_lstm",type=int,default=64)
# parser.add_argument('--emotion_input_dim',type=int,default=6)
# parser.add_argument('--output_dim',type=int,default=3)    
# parser.add_argument('--Max_Comment_Number',type=int,default=193)   
# parser.add_argument('--MAX_sentences',type=int,default=9)   

# parser.add_argument("--h1",type=int,default=256)
# parser.add_argument("--h2",type=int,default=256)
# parser.add_argument('--h3',type=int,default=32)
# parser.add_argument('--h4',type=int,default=64)    
# parser.add_argument('--h5',type=int,default=64)


# parser.add_argument("--learning_rate",type=float,default=9e-4)
# parser.add_argument('--weight_decay',type=float,default=0)
# parser.add_argument("--batch_size",type=int,default=8)
# parser.add_argument('--num_epochs',type=int,default=100)
# parser.add_argument('--patience',type=int,default=20)
# parser.add_argument("--dataset",type=str,default="RumourEval-19")
# args = parser.parse_args()


############################Weibo-comp##############################################
parser.add_argument("--semantic_input_dim",type=int,default=768)
parser.add_argument("--hidden_lstm",type=int,default=64)
parser.add_argument('--emotion_input_dim',type=int,default=9)
parser.add_argument('--output_dim',type=int,default=2)
parser.add_argument('--Max_Comment_Number',type=int,default=45)
parser.add_argument('--MAX_sentences',type=int,default=63)

parser.add_argument("--h1",type=int,default=256)
parser.add_argument("--h2",type=int,default=256)
parser.add_argument('--h3',type=int,default=32)
parser.add_argument('--h4',type=int,default=64)
parser.add_argument('--h5',type=int,default=64)


parser.add_argument("--learning_rate",type=float,default=4e-5)
parser.add_argument('--weight_decay',type=float,default=0)
parser.add_argument("--batch_size",type=int,default=64)
parser.add_argument('--num_epochs',type=int,default=100)
parser.add_argument('--patience',type=int,default=10)
parser.add_argument("--dataset",type=str)
args = parser.parse_args()