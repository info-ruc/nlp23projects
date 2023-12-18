### 代码结构

```
.
├── code
│   ├── data_processor.py	# 下载处理训练数据集
│   ├── model.py			# paddlenlp提供的网络模型
│   ├── predict.py			# 预测代码
│   ├── train.py			# 训练代码
│   └── utils.py			# 通用辅助函数
├── dataset
│   ├── pinyin_vocab.txt	# 拼音
│   └── train.txt			# 训练数据集
├── output
│   ├── model				# 保存模型
│   └── params				# 保存训练得到的参数
└── README.md
```

### 运行方法

#### 数据收集及处理

该模型在SIGHAN简体版数据集以及[Automatic Corpus Generation生成的中文纠错数据集](https://github.com/wdimmy/Automatic-Corpus-Generation/blob/master/corpus/train.sgml)上进行Finetune训练，训练数据集来源于 https://github.com/wdimmy/Automatic-Corpus-Generation/raw/master/corpus/train.sgml

由于拓展训练数据集是.sgml文件，需要将其处理成句子对[err_s \t cor_s]格式才能使用

```sh
python data_processor.py
```

不过本人已经执行过该步骤同时对数据集进行了保留，因此可执行可不执行

#### 模型训练

##### 模型参数

- `max_seq_length` 表示最大句子长度，超过该长度的部分将被切分成下一个样本。
- `batch_size` 表示每次迭代的样本数目。
- `learning_rate` 表示基础学习率大小，将于learning rate scheduler产生的值相乘作为当前学习率。
- `epochs` 表示训练轮数。

##### 简单执行

```sh
python train.py
```

默认参数为batch_size=32   epochs=10   learning_rate=5e-5  max_seq_length=128

##### 自定义参数设置

```sh
python train.py --batch_size 32 \
				--epochs 10 \
				--learning_rate 5e-5 \
				--max_seq_length 128
```

#### 预测

直接运行predict.py文件即可进行拼写纠错检查

```sh
python predict.py
```

由于模型较大且个人本地环境较差，因此训练时间非常非常长，在个人本地训练需要不间断训练2个月的时间，因此并未运行完整，只训练出了一个非常简陋的参数组，效果并不是十分理想(非常非常差，大部分情况都不能进行准确的修改，基本无法使用，由于参数文件非常大且效果不佳，未放入src中，但整个训练框架是完全正确的)，但是由于paddlenlp已经训练出了较好的参数组，且表现很好，足以证实该模型的可行性以及优越性

可以通过下列方法在paddlenlp训练出的参数组上进行预测检查

```python
from paddlenlp import Taskflow
text_correction = Taskflow("text_correction")
text_correction('牛顿顿顿炖牛肉')
```

