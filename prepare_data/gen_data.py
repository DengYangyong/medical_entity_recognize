#encoding=utf8
import os,jieba,csv,random,re
import jieba.posseg as psg
from max_seg import PsegMax

""" 医疗实体词典, 每一行类似：(视力减退,SYM) """
dict_path = "medical_ner_dict.csv"
psgMax = PsegMax(dict_path)

c_root = os.getcwd() + os.sep + "source_data" + os.sep

""" 实体类别 """
biaoji = set(['DIS', 'SYM', 'SGN', 'TES', 'DRU', 'SUR', 'PRE', 'PT', 'Dur', 'TP', 'REG', 'ORG', 'AT', 'PSB', 'DEG', 'FW','CL'])

""" 句子结尾符号，表示如果是句末，则换行 """
fuhao = set(['。','?','？','!','！'])


def add_entity(dict_path):
    """
    把实体字典加载到jieba里，
    实体作为分词后的词，
    实体标记作为词性
    """
    
    dics = csv.reader(open(dict_path,'r',encoding='utf8'))
    
    for row in dics:
        
        if len(row)==2:
            jieba.add_word(row[0].strip(),tag=row[1].strip())
            
            """ 保证由多个词组成的实体词，不被切分开 """
            jieba.suggest_freq(row[0].strip())
            

def split_dataset():
    """
    划分数据集，按照7:2:1的比例
    """
    
    file_all = []
    for file in os.listdir(c_root):
        if "txtoriginal.txt" in file:
            file_all.append(file)
            
    random.seed(10)       
    random.shuffle(file_all)
    
    num = len(file_all)
    train_files = file_all[: int(num * 0.7)]
    dev_files = file_all[int(num * 0.7):int(num * 0.9)]
    test_files = file_all[int(num * 0.9):]
    
    return train_files,dev_files,test_files
               

def sentence_seg(sentence,mode="jieba"):
    """
    1: 实体词典+jieba词性标注。mode="jieba"
    2: 实体词典+双向最大匹配。mode="max_seg"
    """
    
    if mode == "jieba": return psg.cut(sentence)
    if mode == "max_seg": return psgMax.max_biward_seg(sentence)


def auto_label(files, data_type, mode="jieba"):
    """
    不是实体，则标记为O，
    如果是句号等划分句子的符号，则再加换行符，
    是实体，则标记为BI。
    """
    
    writer = open("example.%s" % data_type,"w",encoding="utf8")
    
    for file in files:
        fp = open(c_root+file,'r',encoding='utf8')
        
        for line in fp:
            
            """ 按词性分词 """
            words = sentence_seg(line,mode)
            for word,pos in words: 
                word,pos = word.strip(), pos.strip()   
                if not (word and pos):
                    continue
                
                """ 如果词性不是实体的标记，则打上O标记 """
                if pos not in biaoji:
                   
                    for char in word:
                        string = char + ' ' + 'O' + '\n'
                        
                        """ 在句子的结尾换行 """
                        if char in fuhao:
                            string += '\n'
                            
                        writer.write(string)
                        
                else:
                    
                    """ 如果词性是实体的标记，则打上BI标记"""   
                    begin = 0
                    for char in word:
                        
                        if begin == 0:
                            begin += 1
                            string = char + ' ' + 'B-' + pos + '\n'
                            
                        else:
                            string = char + ' ' + 'I-' + pos + '\n'
                            
                        writer.write(string)
                
    writer.close()
        
def main():
    
    """ 1: 加载实体词和标记到jieba """
    add_entity(dict_path)    
    
    """ 2: 划分数据集 """
    trains, devs, tests = split_dataset()
    
    """ 3: 自动标注样本 """
    for files, data_type in zip([trains,devs,tests],["train","dev","test"]):
        auto_label(files, data_type,mode="max_seg")
        
if __name__ == "__main__":
    
    main()