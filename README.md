# medical_entity_recognize
用BILSTM+CRF做医疗实体识别，框架为pytorch。

注意，代码已经修改了!!

由于pytorch-crf这个包在计算loss时会自动加上<start>和<end>的转移矩阵，所以我们不用再自己手动在样本和标签前后加这俩标记。

然后评估方法改为了CoNLL-2000的权威评估方法。

### 一：环境

    python==3.6.2
    torch==1.1.0
    jieba==0.42.1

### 一：样本自动标注

提供了100篇电子病历和一个医疗实体词典，使用两种方法对样本进行自动标注：

    1: 实体词典+jieba词性标注
    2: 实体词典+双向最大匹配

数据和代码在./prepare_date目录下，运行 gen_date.py 即可。

### 二：命名实体识别项目

上面那一部分和这个项目是独立的，因为100篇电子病历的数据还是比较小，我另外提供了40M左右的医疗NER数据集，已经标注好了，并划分为了训练集/验证集/测试集。

训练集/验证集/测试集的样本量（一个句子为一个样本）为：101218 / 7827 / 16804.

直接运行 sh start.sh 进行训练，训练过程中会完成验证和测试。

    python main.py \
      --train True \
      --hidden_dim=256 \
      --batch_size=128

预测时，修改为 --train False ，再运行 sh start.sh

### 三：模型的效果

模型在测试集上的F1值可达0.976，效果比较好。

### 四：代码的优化点

主要有以下4点:

1 样本和标签前后不需加入<start>和<end>标记，因为pytorch-crf这个包自动会加上这两个标记的转移概率

2 加入了分词特征，做成20维的嵌入，和100维字向量拼接

3 batch分桶，减少zero pad

4 在计算loss时对pad进行mask

5 用CoNLL-2000的评估脚本来评估，权威

### 五：参考代码

BILSTM+CRF的模型主要参考了以下代码，感谢作者：

https://github.com/Alic-yuan/nlp-beginner-finish

CoNLL-2000的python版评估脚本来自：

https://github.com/spyysalo/conlleval.py

### 六：项目文档

《电子病历实体识别》为项目的说明文档

### 七：待完成

1 IDCNN+CRF的模型

2 BERT+CRF的模型
