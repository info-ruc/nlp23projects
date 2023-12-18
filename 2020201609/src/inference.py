from argparse import ArgumentParser

import torch
from tqdm import tqdm
from transformers import (
    MBart50TokenizerFast,
    MBartForConditionalGeneration,
)


def paraphrase(text, model, tokenizer):
    texts = [text] if isinstance(text, str) else text
    inputs = tokenizer(texts, return_tensors="pt", padding=True)["input_ids"].to(model.device)
    result = model.generate(
        inputs,
        num_return_sequences=1,
        do_sample=False,
        temperature=1.0,
        repetition_penalty=10.0,
        max_length=inputs.shape[1] + 10,
        min_length=int(0.5 * (inputs.shape[1] + 10)),
        num_beams=5
    )
    texts = [tokenizer.decode(r, skip_special_tokens=True) for r in result]

    return texts[0]


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument(
        "--model_path", type=str, required=True, help="Specify path to saved model",
    )
    args = parser.parse_args()

    test_data = (
        open("dataset/english_data/test_toxic_parallel.txt", "r").read().split("\n")
    )

    assert len(test_data) == 671

    print(f"Loaded test data")

    model = (
        MBartForConditionalGeneration.from_pretrained(f"{args.model_path}")
        .eval()
        .to(torch.device("cuda"))
    )
    tokenizer = MBart50TokenizerFast.from_pretrained(f"{args.model_path}")

    print(f"Loaded {args.model_path} model")

    result = []
    for sentence in tqdm(test_data):
        out = paraphrase(sentence, model, tokenizer)
        result.append(out)

    with open(f"{args.model_path}/test_results.txt", "w") as f:
        f.write("\n".join(x for x in result))
