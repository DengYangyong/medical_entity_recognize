#coding:utf-8
import math
import random
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BatchManager(object):

    def __init__(self, data,  batch_size):
        self.batch_data = self.sort_and_pad(data, batch_size)
        self.len_data = len(self.batch_data)

    def sort_and_pad(self, data, batch_size):
        """ 
        把样本按长度排序，然后分batch，再pad
        batch之间的输入长度不同，可以减少zero pad，加速计算
        """
        
        num_batch = int(math.ceil(len(data) / batch_size))
        sorted_data = sorted(data, key=lambda x: len(x[0]))
        
        batch_data = list()
        for i in range(num_batch):
            
            """ 进行zero pad """
            batch_data.append(self.pad_data(
                sorted_data[i*int(batch_size): (i+1)*int(batch_size)])
            )
            
        return batch_data

    @staticmethod
    def pad_data(data):
        """
        构造一个mask矩阵，对pad进行mask，不参与loss的计算
        另外，除了id以外，字本身，因为用CoNLL-2000的脚本评估时，需要。
        """
        
        batch_chars = []
        batch_chars_idx = []
        batch_segs_idx = []
        batch_tags_idx = []
        batch_mask = []
        
        max_length = max([len(sentence[0]) for sentence in data])
        for line in data:
            chars, chars_idx, segs_idx, tags_idx = line
            
            padding = [0] * (max_length - len(chars_idx))
            
            """ CoNLL-2000的评估脚本需要"""
            batch_chars.append(chars + padding)
            
            batch_chars_idx.append(chars_idx + padding)
            batch_segs_idx.append(segs_idx + padding)
            batch_tags_idx.append(tags_idx + padding)
            batch_mask.append([1] * len(chars_idx) + padding)
            
        batch_chars_idx = torch.LongTensor(batch_chars_idx).to(device)
        batch_segs_idx = torch.LongTensor(batch_segs_idx).to(device)
        batch_tags_idx = torch.LongTensor(batch_tags_idx).to(device)
        batch_mask = torch.tensor(batch_mask,dtype=torch.uint8).to(device)
               
        return [batch_chars, batch_chars_idx, batch_segs_idx, batch_tags_idx, batch_mask]

    def iter_batch(self, shuffle=True):
        
        if shuffle:
            random.shuffle(self.batch_data)
            
        for idx in range(self.len_data):
            yield self.batch_data[idx]