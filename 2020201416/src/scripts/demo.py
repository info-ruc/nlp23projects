import gradio as gr
import torch
from data import MyDataset
from model import MyModel
from torch.utils.data import DataLoader

args = {
    'sentiments' : {
        0: "Bearish",
        1: "Bullish",
        2: "Neutral"
        },
    'bert_model' : 'bert-base-uncased',
    'max_len' : 128,
    'batch_size' : 32,
    'epochs' : 10,
    'learn_rate' : 1e-5,
    'num_filters' : 768,
    'save_model' : './model/google_model_v1.pth',
    # 'save_model_last' : '/content/drive/MyDrive/FinGPT/model/last_model.pth',
}
model = MyModel(args).to('cpu')
model.load_state_dict(torch.load(args['save_model'], map_location=torch.device('cpu')))

def transfer(Input):
    test_dataset = MyDataset([Input], [0], args)
    test_dataloader = DataLoader(test_dataset, batch_size=args['batch_size'], shuffle=False)
    for batch_text, batch_label in test_dataloader:
        predict = model(batch_text)
        predict = torch.argmax(predict, dim=1).cpu().numpy().tolist()
    return args['sentiments'][predict[0]]

demo = gr.Interface(
    fn=transfer,
    # 设置输入
    inputs=gr.Textbox(label="Input sentence here"),
    # 设置输出
    outputs=gr.TextArea(label="Sentiments"),
    # 设置输入参数示例
    examples=[
        ["Lumentum initiated as positive at Susquehanna"],
        ["I like machine learning."],
        ["According to Gran , the company has no plans to move all production to Russia , although that is where the company is growing ."],
        ["Barclays cuts to Equal Weight"],
        ["The company is growing."],
        ["Apple Just Put the Swiss Watch Industry to Shame."],
        ["Apple embarrasses the Swiss watch industry."],
    ],
    # 设置网页标题
    title="FinBert: Sentiment Classification of Financial News",
    # 左上角的描述文字
    description="Here's a web predicting the sentiment of financial news. Enjoy!",
    # 左下角的文字
    article = "Check out the examples"
)
demo.launch(share=False)
