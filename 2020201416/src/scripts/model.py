import torch.nn as nn
from transformers import BertModel
import torch


class MyModel(nn.Module):
    def __init__(self, args):
        super(MyModel, self).__init__()
        self.args = args
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.bert = BertModel.from_pretrained(self.args['bert_model'])
        # 让 bert 模型进行微调（参数在训练过程中变化）
        for param in self.bert.parameters():
            param.requires_grad = True
        # 全连接层
        self.linear = nn.Linear(self.args['num_filters'], len(self.args['sentiments']))

    def forward(self, x):
        input_ids, attention_mask = x[0].to(self.device), x[1].to(self.device)
        hidden_out = self.bert(input_ids, attention_mask=attention_mask,
                               output_hidden_states=False)  # 控制是否输出所有encoder层的结果
        pred = self.linear(hidden_out.pooler_output)
        return pred