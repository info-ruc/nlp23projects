from functools import partial
import argparse
import os
import random
import time

import numpy as np
import paddle
import paddle.nn.functional as F

import paddlenlp as ppnlp
from paddlenlp.data import Stack, Tuple, Pad, Vocab
from paddlenlp.datasets import load_dataset, MapDataset
from paddlenlp.transformers import LinearDecayWithWarmup
from paddlenlp.transformers import ErnieModel, ErnieTokenizer
from paddlenlp.utils.log import logger
from paddlenlp.metrics import DetectionF1, CorrectionF1
from model import ErnieForCSC
from utils import convert_example, create_dataloader, read_train_ds

from paddle.static import InputSpec

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=32, help="Batch size per CPU for training.")
parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate used to train.")
parser.add_argument("--epochs", type=int, default=10, help="Number of epoches for training.")
parser.add_argument("--max_seq_length", type=int, default=128, help="The maximum total input sequence length after SentencePiece tokenization.")

args = parser.parse_args()


@paddle.no_grad()
def evaluate(model, eval_data_loader):
    model.eval()
    det_metric = DetectionF1()
    corr_metric = CorrectionF1()
    for step, batch in enumerate(eval_data_loader, start=1):
        input_ids, token_type_ids, pinyin_ids, det_labels, corr_labels, length = batch
        det_error_probs, corr_logits = model(input_ids, pinyin_ids, token_type_ids)
        det_metric.update(det_error_probs, det_labels, length)
        corr_metric.update(det_error_probs, det_labels, corr_logits, corr_labels, length)

    det_f1, det_precision, det_recall = det_metric.accumulate()
    corr_f1, corr_precision, corr_recall = corr_metric.accumulate()
    logger.info("Sentence-Level Performance:")
    logger.info(
        "Detection  metric: F1={:.4f}, Recall={:.4f}, Precision={:.4f}".format(det_f1, det_recall, det_precision)
    )
    logger.info(
        "Correction metric: F1={:.4f}, Recall={:.4f}, Precision={:.4f}".format(corr_f1, corr_recall, corr_precision)
    )
    model.train()
    return det_f1, corr_f1


def do_train(args):
    random.seed(1)
    np.random.seed(1)
    paddle.seed(1)
    paddle.set_device("cpu")
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()

    pinyin_vocab = Vocab.load_vocabulary("../dataset/pinyin_vocab.txt", unk_token="[UNK]", pad_token="[PAD]")

    tokenizer = ErnieTokenizer.from_pretrained("ernie-1.0-base-zh")
    ernie = ErnieModel.from_pretrained("ernie-1.0-base-zh")

    model = ErnieForCSC(ernie, pinyin_vocab_size=len(pinyin_vocab), pad_pinyin_id=pinyin_vocab[pinyin_vocab.pad_token])

    train_ds, eval_ds = load_dataset("sighan-cn", splits=["train", "dev"])

    ds = load_dataset(read_train_ds, data_path="../dataset/train.txt", splits=["train"], lazy=False)
    train_ds = MapDataset(ds.data)

    det_loss_act = paddle.nn.CrossEntropyLoss(ignore_index=-1, use_softmax=False)
    corr_loss_act = paddle.nn.CrossEntropyLoss(ignore_index=-1, reduction="none")

    trans_func = partial(
        convert_example, tokenizer=tokenizer, pinyin_vocab=pinyin_vocab, max_seq_length=args.max_seq_length
    )
    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # segment
        Pad(axis=0, pad_val=pinyin_vocab.token_to_idx[pinyin_vocab.pad_token]),  # pinyin
        Pad(axis=0, dtype="int64"),  # detection label
        Pad(axis=0, dtype="int64"),  # correction label
        Stack(axis=0, dtype="int64"),  # length
    ): [data for data in fn(samples)]

    train_data_loader = create_dataloader(
        train_ds, mode="train", batch_size=args.batch_size, batchify_fn=batchify_fn, trans_fn=trans_func
    )

    eval_data_loader = create_dataloader(
        eval_ds, mode="eval", batch_size=args.batch_size, batchify_fn=batchify_fn, trans_fn=trans_func
    )

    num_training_steps = len(train_data_loader) * args.epochs
    # num_training_steps = 20
    lr_scheduler = LinearDecayWithWarmup(args.learning_rate, num_training_steps, 0.1)

    logger.info("Total training step: {}".format(num_training_steps))
    # Generate parameter names needed to perform weight decay.
    # All bias and LayerNorm parameters are excluded.
    decay_params = [p.name for n, p in model.named_parameters() if not any(nd in n for nd in ["bias", "norm"])]
    optimizer = paddle.optimizer.AdamW(
        learning_rate=lr_scheduler,
        epsilon=1e-8,
        parameters=model.parameters(),
        weight_decay=0.01,
        apply_decay_param_fun=lambda x: x in decay_params,
    )

    global_steps = 1
    best_f1 = -1
    tic_train = time.time()
    for epoch in range(args.epochs):
        for step, batch in enumerate(train_data_loader, start=1):
            input_ids, token_type_ids, pinyin_ids, det_labels, corr_labels, length = batch
            det_error_probs, corr_logits = model(input_ids, pinyin_ids, token_type_ids)
            # Chinese Spelling Correction has 2 tasks: detection task and correction task.
            # Detection task aims to detect whether each Chinese charater has spelling error.
            # Correction task aims to correct each potential wrong charater to right charater.
            # So we need to minimize detection loss and correction loss simultaneously.
            det_loss = det_loss_act(det_error_probs, det_labels)
            corr_loss = corr_loss_act(corr_logits, corr_labels) * det_error_probs.max(axis=-1)
            loss = (det_loss + corr_loss).mean()

            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.clear_grad()
            logging_steps = 100 # 日志打印间隔步数
            if global_steps % logging_steps == 0:
                logger.info(
                    "global step %d, epoch: %d, batch: %d, loss: %f, speed: %.2f step/s"
                    % (global_steps, epoch, step, loss, logging_steps / (time.time() - tic_train))
                )
                tic_train = time.time()
            save_steps = 100 # 模型保存及评估间隔步数
            if global_steps % save_steps == 0:
                if paddle.distributed.get_rank() == 0:
                    logger.info("Eval:")
                    det_f1, corr_f1 = evaluate(model, eval_data_loader)
                    f1 = (det_f1 + corr_f1) / 2
                    model_file = "model_%d" % global_steps
                    if f1 > best_f1:
                        # save best model
                        paddle.save(model.state_dict(), os.path.join("../output/params/", "best_model.pdparams"))
                        logger.info("Save best model at {} step.".format(global_steps))
                        best_f1 = f1
                        model_file = model_file + "_best"
                    model_file = model_file + ".pdparams"
                    paddle.save(model.state_dict(), os.path.join("../output/params/", model_file))
                    logger.info("Save model at {} step.".format(global_steps))
            if global_steps >= num_training_steps:
                return
            global_steps += 1

def export_model(): # 根据训练出的参数导出静态模型用于后续的预测
    pinyin_vocab = Vocab.load_vocabulary("../dataset/pinyin_vocab.txt", unk_token="[UNK]", pad_token="[PAD]")
    ernie = ErnieModel.from_pretrained("ernie-1.0-base-zh")
    model = ErnieForCSC(ernie, pinyin_vocab_size=len(pinyin_vocab), pad_pinyin_id=pinyin_vocab[pinyin_vocab.pad_token])

    model_dict = paddle.load("../output/params/best_model.pdparams")
    model.set_dict(model_dict)
    model.eval()

    model = paddle.jit.to_static(
        model,
        input_spec=[
            InputSpec(shape=[None, None], dtype="int64", name="input_ids"),
            InputSpec(shape=[None, None], dtype="int64", name="pinyin_ids"),
        ],
    )
    paddle.jit.save(model, "../output/model/static_graph_params")

if __name__ == "__main__":
    do_train(args)
    export_model()

