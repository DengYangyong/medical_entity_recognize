import torch
import torch.nn as nn
import torch.optim as optim
from model.torchcrf import CRF

class NERLSTM_CRF(nn.Module):
    def __init__(self, config, char2id, tag2id, emb_matrix):
        super(NERLSTM_CRF, self).__init__()

        self.hidden_dim = config.hidden_dim
        self.vocab_size = len(char2id)
        self.seg_size = 5
        self.tag_to_ix = tag2id
        self.tagset_size = len(tag2id)
        
        """ pdding_idx=0，也就是让pad标记不更新 """
        self.char_emb = nn.Embedding.from_pretrained(
            emb_matrix,freeze=False, padding_idx=0
        )
        self.seg_emb = nn.Embedding(
            self.seg_size, config.seg_dim, padding_idx=0
        )
        self.emb_dim = config.char_dim + config.seg_dim
        
        self.dropout = nn.Dropout(config.dropout)
        self.lstm = nn.LSTM(
            self.emb_dim, self.hidden_dim // 2, num_layers=1, 
            bidirectional=True, batch_first=True
        )
        
        """ 得到发射矩阵 """
        self.hidden2tag = nn.Linear(self.hidden_dim, self.tagset_size)
        
        self.crf = CRF(self.tagset_size,batch_first=True)

    def forward(self, char_ids,seg_ids,mask=None):
        
        """ 把字向量（100维）和词长度特征向量（20维），拼接 """
        embedding = torch.cat(
            (self.char_emb(char_ids),self.seg_emb(seg_ids)), 2
        )
        
        outputs, hidden = self.lstm(embedding)
        outputs = self.dropout(outputs)
        outputs = self.hidden2tag(outputs)
        
        """ 预测时，得到维特比解码的路径 """
        return self.crf.decode(outputs, mask)

    def log_likelihood(self, char_ids, seg_ids, tag_ids, mask=None):
        
        embedding = torch.cat(
            (self.char_emb(char_ids),self.seg_emb(seg_ids)), 2
        )
 
        outputs, hidden = self.lstm(embedding)
        outputs = self.dropout(outputs)
        outputs = self.hidden2tag(outputs)
        
        """ 训练时，得到损失 """
        return - self.crf(outputs, tag_ids, mask)
