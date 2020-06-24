#coding:utf-8
from data_utils import char_mapping,tag_mapping,augment_with_pretrained
from data_utils import zero_digits,iob, iob_iobes, get_seg_features
from logs.logger import logger
from params import params
import os
import pickle
from tqdm import tqdm
import numpy as np
import torch

config = params()


def build_dataset():
    
    train_sentences = load_sentences(
        config.train_file, config.lower, config.zero
    )
    dev_sentences = load_sentences(
        config.dev_file, config.lower, config.zero
    )
    test_sentences = load_sentences(
        config.test_file, config.lower, config.zero
    )
    logger.info("成功读取标注好的数据")


    update_tag_scheme(
        train_sentences, config.tag_schema
    )
    update_tag_scheme(
        test_sentences, config.tag_schema
    )
    update_tag_scheme(
        dev_sentences, config.tag_schema
    )
    logger.info("成功将IOB格式转化为IOBES格式")
    
    
    if not os.path.isfile(config.map_file):
        char_to_id, id_to_char, tag_to_id, id_to_tag = create_maps(train_sentences)
        logger.info("根据训练集建立字典完毕")
    else:
        with open(config.map_file, "rb") as f:
            char_to_id, id_to_char, tag_to_id, id_to_tag = pickle.load(f)
        logger.info("已有字典文件，加载完毕")
            
            
    emb_matrix = load_emb_matrix(char_to_id)
    logger.info("加载预训练的字向量完毕")
            
            
    train_data = prepare_dataset(
        train_sentences, char_to_id, tag_to_id, config.lower
    )
    dev_data = prepare_dataset(
        dev_sentences, char_to_id, tag_to_id, config.lower
    )
    test_data = prepare_dataset(
        test_sentences, char_to_id, tag_to_id, config.lower
    )
    logger.info("把样本和标签处理为id完毕")
    logger.info("%i / %i / %i sentences in train / dev / test." % (
        len(train_data), len(dev_data), len(test_data))
    ) 
    
    with open(config.data_proc_file, "wb") as f:
        pickle.dump([train_data,dev_data,test_data], f)
        pickle.dump([char_to_id,id_to_char,tag_to_id,id_to_tag], f)
        pickle.dump(emb_matrix, f)
        
    return train_data,dev_data,test_data, char_to_id, tag_to_id, id_to_tag, emb_matrix
    

def load_sentences(path, lower, zero):
    """
    加载训练样本，一句话就是一个样本。
    训练样本中，每一行是这样的：长 B-Dur，即字和对应的标签
    句子之间使用空行隔开的
    return : sentences: [[[['无', 'O'], ['长', 'B-Dur'], ['期', 'I-Dur'],...]]
    """
    
    sentences = []
    sentence = []
   
    for line in open(path, 'r',encoding='utf8'):
        
        """ 如果包含有数字，就把每个数字用0替换 """
        line = line.rstrip()
        line = zero_digits(line) if zero else line
        
        """ 如果不是句子结束的换行符，就继续添加单词到句子中 """
        if line:
            word_pair = ["<unk>", line[2:]] if line[0] == " " else line.split()
            assert len(word_pair) == 2
            sentence.append(word_pair)     
             
        else:
            
            """ 如果遇到换行符，说明一个句子处理完毕 """
            if len(sentence) > 0:
                sentences.append(sentence)
                sentence = []
                
    """ 最后一个句子没有换行符，处理好后，直接添加到样本集中 """   
    if len(sentence) > 0:
        sentences.append(sentence)
        
    return  sentences


def update_tag_scheme(sentences, tag_scheme):
    """
    1：检查样本的标签是否为正确的IOB格式，如果不对则纠正。
    2：将IOB格式转化为IOBES格式。
    """
    
    for i, s in enumerate(sentences):
        
        tags = [w[-1] for w in s]
       
        if not iob(tags):
            s_str = '\n'.join(' '.join(w) for w in s)
            print('Sentences should be given in IOB format! \n' +
                  'Please check sentence %i:\n%s' % (i, s_str))
        
        """ 如果用IOB格式训练，则检查并纠正一遍 """
        if tag_scheme == 'iob':
            
            for word, new_tag in zip(s, tags):
                word[-1] = new_tag
        
        elif tag_scheme == 'iobes':
            
            """ 将IOB格式转化为IOBES格式 """
            new_tags = iob_iobes(tags)
            for word, new_tag in zip(s, new_tags):
                word[-1] = new_tag
    

def create_maps(sentences):
    """
    建立字和标签的字典
    """
    
    if config.pre_emb:
        
        """ 首先利用训练集建立字典 """
        dico_chars_train, _, _ = char_mapping(sentences, config.lower)
        
        """ 预训练字向量中的字，如果不在上面的字典中，则加入 """
        dico_chars, char_to_id, id_to_char = augment_with_pretrained(dico_chars_train.copy(),
                                                                     config.emb_file)
        
    else:
        
        """ 只利用训练集建立字典 """
        _, char_to_id, id_to_char = char_mapping(sentences, config.lower)
    
    """ 利用训练集建立标签字典 """
    _, tag_to_id, id_to_tag = tag_mapping(sentences)
                   
    with open(config.map_file, "wb") as f:
        pickle.dump([char_to_id, id_to_char, tag_to_id, id_to_tag], f)
        
    return char_to_id, id_to_char, tag_to_id, id_to_tag


def prepare_dataset(sentences, char_to_id, tag_to_id, lower=False, test=False):
    
    """
    把文本型的样本和标签，转化为index，便于输入模型
    需要在每个样本和标签前后加<start>和<end>,
    但由于pytorch-crf这个包里面会自动添加<start>和<end>的转移概率，
    所以我们不用在手动加入。
    """

    def f(x): return x.lower() if lower else x
    
    data = []
    for s in sentences:
        
        chars = [w[0] for w in s]
        tags = [w[-1] for w in s]
        
        """ 句子转化为index """
        chars_idx = [char_to_id[f(c) if f(c) in char_to_id else '<unk>'] for c in chars]
        
        """ 对句子分词，构造词的长度特征 """
        segs_idx = get_seg_features("".join(chars))
        
        if not test:
            tags_idx =  [tag_to_id[t] for t in tags]
            
        else:
            tags_idx = [tag_to_id["<pad>"] for _ in tags]
            
        assert len(chars_idx) == len(segs_idx) == len(tags_idx)
        data.append([chars, chars_idx, segs_idx, tags_idx])

    return data

""" 加载预训练字向量，并与词表相对应 """ 
def load_emb_matrix(vocab):
    
    """ 1: 加载字向量 """ 
    print("\nLoading char2vec ...\n")
    emb_index = load_w2v(config.emb_file)
    
    """ 2: 字向量矩阵与词表相对应 """ 
    vocab_size = len(vocab)
    emb_matrix = np.zeros((vocab_size,config.char_dim))
    for word,index in vocab.items():
        vector = emb_index.get(word)
        if vector is not None:
            emb_matrix[index] = vector
            
    emb_matrix = torch.FloatTensor(emb_matrix)
            
    return emb_matrix
        
""" 字向量 """
def load_w2v(path):
    
    file = open(path,encoding="utf-8")
    
    emb_idx = {}
    for i,line in tqdm(enumerate(file)):
        value = line.split()
        char = value[0]
        emb = np.asarray(value[1:], dtype="float32")
        if len(emb) != config.char_dim: continue
        emb_idx[char] = emb
        
    return emb_idx

if __name__ == "__main__":
    
    build_dataset()
