#coding:utf-8
import argparse
import os
import torch,random
import numpy as np

root_path = os.getcwd() + os.sep

""" 设置随机数种子 """
def set_manual_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
set_manual_seed(20)
print("设置随机数种子为20")

def str2bool(str):
    return True if str.lower() == 'true' else False

def params():
    parser = argparse.ArgumentParser()
    add_arg = parser.add_argument
    
    add_arg("--train", default=True, help="Whether train the model",type=str2bool)
    
    add_arg("--seg_dim",default=20, help="Embedding size for segmentation, 0 if not used", type=int)
    add_arg("--char_dim", default=100, help="Embedding size for characters", type=int)
    add_arg("--hidden_dim", default=256, help="Num of hidden units in LSTM", type=int)
    add_arg("--tag_schema", default="iobes", help="tagging schema iobes or iob", type=str)
    
    add_arg("--clip", default=5, help="Gradient clip", type=float)
    add_arg("--dropout", default=0.5, help="Dropout rate", type=float)
    add_arg("--batch_size", default=8, help="batch size", type=int)
    add_arg("--lr", default=0.003, help="Initial learning rate", type=float)
    add_arg("--weight_decay", default=1e-5, help="Learning rate decay", type=float)
    add_arg("--optimizer", default="adam", help="Optimizer for training",type=str)
    add_arg("--pre_emb", default=True, help="Wither use pre-trained embedding",type=str2bool)
    add_arg("--zero", default=True, help="Wither replace digits with zero",type=str2bool)
    add_arg("--lower", default=True, help="Wither lower case",type=str2bool)
    
    add_arg("--max_epoch", default=50, help="maximum training epochs",type=int)
    add_arg("--steps_check", default=100, help="steps per checkpoint",type=int)
    add_arg("--save_dir", default=os.path.join(root_path,"result"), help="Path to save model",type=str)
    add_arg("--log_file", default="train.log", help="File for log",type=str)
    add_arg("--map_file", default=os.path.join(root_path+"data","maps.pkl"), help="file for maps",type=str)
    add_arg("--data_proc_file", default=os.path.join(root_path+"data","data_proc.pkl"), help="file for processed data",type=str)
    add_arg("--emb_file", default=os.path.join(root_path+"data", "vec.txt"), help="Path for pre_trained embedding",type=str)
    
    add_arg("--train_file", default=os.path.join(root_path+"data", "example.train"), help="Path for train data",type=str)
    add_arg("--dev_file", default=os.path.join(root_path+"data", "example.dev"), help="Path for dev data",type=str)
    add_arg("--test_file", default=os.path.join(root_path+"data", "example.test"), help="Path for test data",type=str)
    
    add_arg("--model_type", default="bilstm", help="Model type, can be idcnn or bilstm",type=str)
    add_arg("--require_improve", default=5000, help="Max step for early stop",type=int)
    args = parser.parse_args()
    return args
