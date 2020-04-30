#coding:utf-8
import pandas as pd

class PsegMax:
    def __init__(self, dict_path):
        self.entity_dict, self.max_len = self.load_entity(dict_path)

    def load_entity(self, dict_path):
        """
        加载实体词典
        """
        
        entity_list = []
        max_len = 0
        
        """ 实体词典: {'肾抗针': 'DRU', '肾囊肿': 'DIS', '肾区': 'REG', '肾上腺皮质功能减退症': 'DIS', ...} """
        df = pd.read_csv(dict_path,header=None,names=["entity","tag"])
        entity_dict = {entity.strip(): tag.strip() for entity,tag in df.values.tolist()}
        
        """ 计算词典中实体的最大长度 """
        df["len"] = df["entity"].apply(lambda x:len(x))
        max_len = max(df["len"])
        
        return entity_dict, max_len

    
    def max_forward_seg(self, sent):
        """
        前向最大匹配实体标注
        """
        words_pos_seg = []
        sent_len = len(sent)
        
        while sent_len > 0:
            
            """ 如果句子长度小于实体最大长度，则切分的最大长度为句子长度 """
            max_len = min(sent_len,self.max_len)
            
            """ 从左向右截取max_len个字符，去词典中匹配 """
            sub_sent = sent[:max_len]
            
            while max_len > 0:
                
                """ 如果切分的词在实体词典中，那就是切出来的实体 """
                if sub_sent in self.entity_dict:
                    tag = self.entity_dict[sub_sent]
                    words_pos_seg.append((sub_sent,tag))
                    break
                
                elif max_len == 1:
                    
                    """ 如果没有匹配上，那就把单个字切出来，标签为O """
                    tag = "O"
                    words_pos_seg.append((sub_sent,tag))
                    break
                
                else:
                    
                    """ 如果没有匹配上，又还没剩最后一个字，就去掉右边的字,继续循环 """
                    max_len -= 1
                    sub_sent = sub_sent[:max_len]
            
            """ 把分出来的词（实体或单个字）去掉，继续切分剩下的句子 """
            sent = sent[max_len:]
            sent_len -= max_len
     
        return words_pos_seg

    
    def max_backward_seg(self, sent):
        """
        后向最大匹配实体标注
        """

        words_pos_seg = []
        sent_len = len(sent)
        
        while sent_len > 0:
            
            """ 如果句子长度小于实体最大长度，则切分的最大长度为句子长度 """
            max_len = min(sent_len,self.max_len)
            
            """ 从右向左截取max_len个字符，去词典中匹配 """
            sub_sent = sent[-max_len:]
            
            while max_len > 0:
                
                """ 如果切分的词在实体词典中，那就是切出来的实体 """
                if sub_sent in self.entity_dict:
                    tag = self.entity_dict[sub_sent]
                    words_pos_seg.append((sub_sent,tag))
                    break
                
                elif max_len == 1:
                    
                    """ 如果没有匹配上，那就把单个字切出来，标签为O """
                    tag = "O"
                    words_pos_seg.append((sub_sent,tag))
                    break
                
                else:
                    
                    """ 如果没有匹配上，又还没剩最后一个字，就去掉右边的字,继续循环 """
                    max_len -= 1
                    sub_sent = sub_sent[-max_len:]
            
            """ 把分出来的词（实体或单个字）去掉，继续切分剩下的句子 """
            sent = sent[:-max_len]
            sent_len -= max_len
        
        """ 把切分的结果反转 """
        return words_pos_seg[::-1]


    def max_biward_seg(self, sent):
        """
        双向最大匹配实体标注
        """
        
        """ 1: 前向和后向的切分结果 """
        words_psg_fw = self.max_forward_seg(sent)
        words_psg_bw = self.max_backward_seg(sent)
        
        """ 2: 前向和后向的词数 """
        words_fw_size = len(words_psg_fw)
        words_bw_size = len(words_psg_bw)
        
        """ 3: 前向和后向的词数，则取词数较少的那个 """
        if words_fw_size < words_bw_size: return words_psg_fw
        
        if words_fw_size > words_bw_size: return words_psg_bw

        """ 4: 结果相同，可返回任意一个 """
        if words_psg_fw == words_psg_bw: return words_psg_fw
        
        """ 5: 结果不同，返回单字较少的那个 """
        fw_single = sum([1 for i in range(words_fw_size) if len(words_psg_fw[i][0])==1])
        bw_single = sum([1 for i in range(words_fw_size) if len(words_psg_bw[i][0])==1])
        
        if fw_single < bw_single: return words_psg_fw
        else: return words_psg_bw
            

if __name__ == "__main__":
    
    dict_path = "medical_ner_dict.csv"
    text = "我最近双下肢疼痛，我该咋办。"
    
    psg = PsegMax(dict_path)
    words_psg = psg.max_biward_seg(text)
    
    print(words_psg)
    
