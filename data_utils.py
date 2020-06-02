# encoding = utf8
import os,re
import math
import codecs
import random
import numpy as np
import jieba
from logs.logger import logger
from conlleval import return_report

def create_dico(item_list):
    """
    统计列表元素的频率，构成一个字典
    """
    assert type(item_list) is list
    dico = {}
    for items in item_list:
        for item in items:
            if item not in dico:
                dico[item] = 1
            else:
                dico[item] += 1
    return dico


def create_mapping(dico):
    """
    建立字和id对应的字典，按频率降序排列
    """
    sorted_items = sorted(dico.items(), key=lambda x: (-x[1], x[0]))
    id_to_item = {i: v[0] for i, v in enumerate(sorted_items)}
    item_to_id = {v: k for k, v in id_to_item.items()}
    return item_to_id, id_to_item


def zero_digits(s):
    """
    把句子中的数字统一用0替换.
    """
    return re.sub('\d', '0', s)


def iob(tags):
    """
    检查tags是否为正确的IOB格式，不正确则纠正。
    """
    for i, tag in enumerate(tags):
        
        if tag == 'O': continue
        split = tag.split('-')
        if len(split) != 2 or split[0] not in ['I', 'B']: return False
        if split[0] == 'B': continue
        elif i == 0 or tags[i - 1] == 'O': tags[i] = 'B' + tag[1:]
        elif tags[i - 1][1:] == tag[1:]: continue
        else: tags[i] = 'B' + tag[1:]
        
    return True


def iob_iobes(tags):
    """
    IOB -> IOBES
    """
    new_tags = []
    
    for i, tag in enumerate(tags):
        if tag == 'O':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'B': 
            if i + 1 != len(tags) and tags[i + 1].split('-')[0] == 'I':
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace('B-', 'S-'))         
        elif tag.split('-')[0] == 'I':    
            if i + 1 < len(tags) and tags[i + 1].split('-')[0] == 'I':
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace('I-', 'E-'))        
        else:
            raise Exception('Invalid IOB format!')       
    return new_tags


def iobes_iob(tags):
    """
    IOBES -> IOB
    """
    new_tags = []
    for i, tag in enumerate(tags):
        if tag.split('-')[0] == 'B':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'I':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'S':
            new_tags.append(tag.replace('S-', 'B-'))
        elif tag.split('-')[0] == 'E':
            new_tags.append(tag.replace('E-', 'I-'))
        elif tag.split('-')[0] == 'O':
            new_tags.append(tag)
        else:
            raise Exception('Invalid format!')
    return new_tags

def char_mapping(sentences, lower):
    """
    建立字和id对应的字典，按频率降序排列
    """
    chars = [[x[0].lower() if lower else x[0] for x in s] for s in sentences]
    dico = create_dico(chars)
    dico["<pad>"] = 100000003
    dico['<unk>'] = 100000002
    
    char_to_id, id_to_char = create_mapping(dico)
    logger.info("Found %i unique words (%i in total)" % (len(dico), sum(len(x) for x in chars)))
    
    return dico, char_to_id, id_to_char


def tag_mapping(sentences):
    """
    建立标签和id对应的字典，按频率降序排列
    由于用了CRF，所以需要在标签前后加<start>和<end>
    但是torchcrf那个包会自动处理，那么在字典中不用加入这两个标记
    """
    
    f = open('data/tag_to_id.txt','w',encoding='utf8')
    f1 = open('data/id_to_tag.txt','w',encoding='utf8')
    
    tags = [[x[-1] for x in s] for s in sentences]
    
    dico = create_dico(tags)
    dico["<pad>"] = 100000002

    tag_to_id, id_to_tag = create_mapping(dico)
    
    logger.info("Found %i unique named entity tags" % len(dico))
    for k,v in tag_to_id.items():
        f.write(k+":"+str(v)+"\n")
    for k,v in id_to_tag.items():
        f1.write(str(k)+":"+str(v)+"\n")
    return dico, tag_to_id, id_to_tag


def augment_with_pretrained(dictionary, ext_emb_path):
    """
    预训练字向量中的字，如果不在训练集的字典中，就加入，拓展字典。
    """
    logger.info('Loading pretrained embeddings from %s...' % ext_emb_path)
    assert os.path.isfile(ext_emb_path)

    """ 加载预训练的字向量 """
    pretrained = set([
        line.rstrip().split()[0].strip()
        for line in codecs.open(ext_emb_path, 'r', 'utf-8')
        if len(ext_emb_path) > 0
    ])
    
    for char in pretrained:
        if char not in dictionary:
            dictionary[char] = 0

    char_to_id, id_to_char = create_mapping(dictionary)
    return dictionary, char_to_id, id_to_char


def get_seg_features(string):
    """
    对句子分词，构造词的长度特征，为BIES格式,
    [对]对应的特征为[4], 不设为0，因为pad的id就是0
    [句子]对应的特征为[1,3],
    [中华人民]对应的特征为[1,2,2,3]
    """
    seg_feature = []

    for word in jieba.cut(string):
        if len(word) == 1:
            seg_feature.append(4)
        else:
            tmp = [2] * len(word)
            tmp[0] = 1
            tmp[-1] = 3
            seg_feature.extend(tmp)
    return seg_feature


def test_ner(results, path):
    """
    用CoNLL-2000的实体识别评估脚本来评估模型
    """
    
    """ 用CoNLL-2000的脚本，需要把预测结果保存为文件，再读取 """
    output_file = os.path.join(path, "ner_predict.utf8")
    with open(output_file, "w",encoding='utf8') as f:
        to_write = []
        for block in results:
            for line in block:
                to_write.append(line + "\n")
            to_write.append("\n")

        f.writelines(to_write)
    eval_lines = return_report(output_file)
    return eval_lines


def result_to_json(string, tags):
    """ 按规范的格式输出预测结果 """
    
    item = {"string": string, "entities": []}
    entity_name = ""
    entity_start = 0
    idx = 0
    for char, tag in zip(string, tags):
        if tag[0] == "S":
            item["entities"].append({"word": char, "start": idx, "end": idx+1, "type":tag[2:]})
        elif tag[0] == "B":
            entity_name += char
            entity_start = idx
        elif tag[0] == "I":
            entity_name += char
        elif tag[0] == "E":
            entity_name += char
            item["entities"].append({"word": entity_name, "start": entity_start, "end": idx + 1, "type": tag[2:]})
            entity_name = ""
        else:
            entity_name = ""
            entity_start = idx
        idx += 1
    return item
