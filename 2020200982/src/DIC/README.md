### 代码结构

```
.
├── code
│   ├── csc.py					# 实现文件
│   └── data_processor.py		# 数据下载及处理
├── dataset						# 数据集
│   ├── cn_dic.txt				
│   ├── frequent_position.txt	
│   ├── train.sgml				
│   └── train.txt				
└── README.md
```

### 运行方法

#### 数据收集及处理

该模型在SIGHAN简体版数据集以及[Automatic Corpus Generation生成的中文纠错数据集](https://github.com/wdimmy/Automatic-Corpus-Generation/blob/master/corpus/train.sgml)上进行pre-train，训练数据集来源于 https://github.com/wdimmy/Automatic-Corpus-Generation/raw/master/corpus/train.sgml

由于拓展训练数据集是.sgml文件，需要将其处理成句子对[err_s \t cor_s]格式才能使用

```sh
python data_processor.py
```

不过本人已经执行过该步骤同时对数据集进行了保留，因此可执行可不执行

#### 预测

由于基于词的方法只需要通过把句子分割成词组，然后对词组进行检查纠正，因此不需要训练的过程，直接执行即可进行拼写纠错

```sh
python csc.py
```

由于基于词的方法可以完成基本的日常纠错任务，优点在于速度快（几乎不花时间）、修改细致、把握性高、可解释性强，但是需要依赖大量的修改语料数据作为支撑，且对于语义错误有时难以修改