import os
import argparse
import numpy as np
from functools import partial

import paddle
from paddle import inference
from paddlenlp.data import Stack, Tuple, Pad, Vocab
from paddlenlp.transformers import ErnieTokenizer
from utils import convert_example, parse_decode


class Predictor(object):
    def __init__(self, model_file, params_file, device, max_seq_length, tokenizer, pinyin_vocab):
        self.max_seq_length = max_seq_length

        config = paddle.inference.Config("../output/model/static_graph_params.pdmodel", 
                                        "../output/model/static_graph_params.pdiparams")
        config.disable_gpu()
        config.switch_use_feed_fetch_ops(False)
        self.predictor = paddle.inference.create_predictor(config)

        self.input_handles = [self.predictor.get_input_handle(name) for name in self.predictor.get_input_names()]

        self.det_error_probs_handle = self.predictor.get_output_handle(self.predictor.get_output_names()[0])
        self.corr_logits_handle = self.predictor.get_output_handle(self.predictor.get_output_names()[1])
        self.tokenizer = tokenizer
        self.pinyin_vocab = pinyin_vocab

    def predict(self, data, batch_size=1):
        """
        Predicts the data labels.

        Args:
            data (obj:`List(Example)`): The processed data whose each element is a Example (numedtuple) object.
                A Example object contains `text`(word_ids) and `seq_len`(sequence length).
            batch_size(obj:`int`, defaults to 1): The number of batch.

        Returns:
            results(obj:`dict`): All the predictions labels.
        """
        examples = []
        texts = []
        trans_func = partial(
            convert_example,
            tokenizer=self.tokenizer,
            pinyin_vocab=self.pinyin_vocab,
            max_seq_length=self.max_seq_length,
            is_test=True,
        )

        batchify_fn = lambda samples, fn=Tuple(
            Pad(axis=0, pad_val=self.tokenizer.pad_token_id, dtype="int64"),  # input
            Pad(axis=0, pad_val=self.tokenizer.pad_token_type_id, dtype="int64"),  # segment
            Pad(axis=0, pad_val=self.pinyin_vocab.token_to_idx[self.pinyin_vocab.pad_token], dtype="int64"),  # pinyin
            Stack(axis=0, dtype="int64"),  # length
        ): [data for data in fn(samples)]

        for text in data:
            example = {"source": text.strip()}
            input_ids, token_type_ids, pinyin_ids, length = trans_func(example)
            examples.append((input_ids, token_type_ids, pinyin_ids, length))
            texts.append(example["source"])

        batch_examples = [examples[idx : idx + batch_size] for idx in range(0, len(examples), batch_size)]
        batch_texts = [texts[idx : idx + batch_size] for idx in range(0, len(examples), batch_size)]
        results = []

        for examples, texts in zip(batch_examples, batch_texts):
            token_ids, token_type_ids, pinyin_ids, length = batchify_fn(examples)
            self.input_handles[0].copy_from_cpu(token_ids)
            self.input_handles[1].copy_from_cpu(pinyin_ids)
            self.predictor.run()
            det_error_probs = self.det_error_probs_handle.copy_to_cpu()
            corr_logits = self.corr_logits_handle.copy_to_cpu()

            det_pred = det_error_probs.argmax(axis=-1)
            char_preds = corr_logits.argmax(axis=-1)

            for i in range(len(length)):
                pred_result = parse_decode(
                    texts[i], char_preds[i], det_pred[i], length[i], self.tokenizer, self.max_seq_length
                )

                results.append("".join(pred_result))
        return results


if __name__ == "__main__":
    tokenizer = ErnieTokenizer.from_pretrained("ernie-1.0")
    pinyin_vocab = Vocab.load_vocabulary("../dataset/pinyin_vocab.txt", unk_token="[UNK]", pad_token="[PAD]")
    predictor = Predictor("../output/model/static_graph_params.pdmodel", # 加载模型
                          "../output/model/static_graph_params.pdiparams", # 加载参数
                          "cpu", # 设置硬件，可选择gpu
                          64, # 单句最大长度
                          tokenizer, pinyin_vocab)
    while True:
        err_sent = input("Please input a sentence:\n")
        s = []
        s.append(err_sent)
        results = predictor.predict(s, batch_size=3)
        for source, target in zip(s, results):
            print("Source:", source)
            print("Target:", target)
