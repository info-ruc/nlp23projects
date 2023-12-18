import argparse
import os

import torch
from transformers import (
    MBart50TokenizerFast,
    MBartForConditionalGeneration,
    Trainer,
    TrainingArguments,
)

from utils import ToxicDataset, load_data, set_random_seed

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument(
        "--max_steps",
        type=int,
        default=5000,
        help="maximum learning steps ([1000, 3000, 5000, 10000])",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="learning rate for fine-tuning ([1e-3, 1e-4, 1e-5, 3e-5])",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=0,
        help="number of warmup steps ([0, 100, 500, 1000])",
    )
    parser.add_argument(
        "--n_device",
        type=int,
        default=0,
        help="num of device",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="mbarts",
        help="set directory for model saving",
    )
    args = parser.parse_args()

    set_random_seed(42)
    os.environ["WANDB_DISABLED"] = "true"
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.n_device}"

    train_data, tune_data = load_data()
    print("data loaded")

    model = MBartForConditionalGeneration.from_pretrained(
        "facebook/mbart-large-50"
    )
    tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50")
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    model.resize_token_embeddings(len(tokenizer))
    device = torch.device("cuda") if torch.cuda.is_available else "cpu"
    model = model.to(device)
    print("model loaded")
    trainset = ToxicDataset(train_data, tokenizer)
    tuneset = ToxicDataset(tune_data, tokenizer)

    train_args = TrainingArguments(
        output_dir=f"{args.output_dir}/mbart_{args.max_steps}_ENG",
        do_train=True,
        do_eval=True,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        evaluation_strategy="steps",
        logging_steps=1000,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        seed=42,
        save_strategy="steps",
        save_steps=5000,
        warmup_steps=args.warmup_steps,
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=trainset,
        eval_dataset=tuneset,
        tokenizer=tokenizer,
    )

    print("training started")
    trainer.train()
    model.save_pretrained(f"{args.output_dir}/mbart_{args.max_steps}_ENG")
    tokenizer.save_pretrained(f"{args.output_dir}/mbart_{args.max_steps}_ENG")
