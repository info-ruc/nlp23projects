def create_answers_dict(x):
    dict_ = {"text":[(x["answer"])], "answer_start":[(int(x["answer_start"]))]}
    return dict_



import pandas as pd

#train
contexts_df_train = pd.DataFrame(train_contexts, columns=['context'])
questions_df_train = pd.DataFrame(train_questions, columns=['question'])
answers_df_train = pd.DataFrame.from_records(train_answers)
df_train = contexts_df_train.copy()
df_train["question"] = questions_df_train["question"]
df_train["answer"] = answers_df_train["text"]
df_train["answer_start"] = answers_df_train["answer_start"]
df_train.reset_index(inplace=True, drop = False)
df_train.rename(columns={'index':'id'}, inplace=True)
df_train["answers"] = df_train.apply(lambda x: create_answers_dict(x), axis = 1)

#test
contexts_df_test = pd.DataFrame(val_contexts, columns=['context'])
questions_df_test = pd.DataFrame(val_questions, columns=['question'])
answers_df_test = pd.DataFrame.from_records(val_answers)
df_test = contexts_df_test.copy()
df_test["question"] = questions_df_test["question"]
df_test["answer"] = answers_df_test["text"]
df_test["answer_start"] = answers_df_test["answer_start"]
df_test.reset_index(inplace=True, drop = False)
df_test.rename(columns={'index':'id'}, inplace=True)
df_test["answers"] =  df_test.apply(lambda x: create_answers_dict(x), axis = 1)
df_test.tail()

df_train.sample(frac = 0.5)[['id', 'context', 'question', 'answers']].to_csv('data/outputs/dataset_train.csv', index=False)
df_test.sample(frac = 0.5)[['id', 'context', 'question', 'answers']].to_csv('data/outputs/dataset_test.csv', index=False)


from datasets import load_dataset

data_files = {"train": "data/outputs/dataset_train.csv", "test": "data/outputs/dataset_test.csv"}
ds = load_dataset("csv", data_files=data_files)
ds


def convert_text(batch):
  aux_list = []
  for x, y in zip(batch["answers"], batch["answers"]):
    my_dict = {"text":eval(x)["text"], "answer_start":eval(x)["answer_start"]}
    aux_list.append(my_dict)

  return {"texts":aux_list}

prepared_ds = ds.map(convert_text, batched = True)
prepared_ds = prepared_ds.remove_columns("answers")
prepared_ds = prepared_ds.rename_column("texts", "answers")


from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def preprocess_function(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=384,
        truncation="only_second",
        return_offsets_mapping=True,
        padding="max_length",
    )

    offset_mapping = inputs.pop("offset_mapping")
    print(offset_mapping)
    answers = examples["answers"]
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        answer = answers[i]
        start_char = answer["answer_start"][0]
        end_char = answer["answer_start"][0] + len(answer["text"][0])
        sequence_ids = inputs.sequence_ids(i)

        # Find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # If the answer is not fully inside the context, label it (0, 0)
        if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Otherwise it's the start and end token positions
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs


    tokenized_squad = prepared_ds.map(preprocess_function, batched=True, remove_columns=ds["train"].column_names)


    from dotenv import load_dotenv
    from huggingface_hub import login

    load_dotenv('./secret/keys.env')
    HUGGING_FACE_API_KEY = os.getenv("HUGGINGFACE_TOKEN")
    login(token = HUGGING_FACE_API_KEY)


    from transformers import DefaultDataCollator, AutoModelForQuestionAnswering, TrainingArguments, Trainer

    data_collator = DefaultDataCollator()

    model = AutoModelForQuestionAnswering.from_pretrained("distilbert-base-uncased")

    training_args = TrainingArguments(
        output_dir="qa_nlp_model",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        push_to_hub=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_squad["train"],
        eval_dataset=tokenized_squad["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()



    trainer.save_model()
    metrics = trainer.evaluate(tokenized_squad["test"])

    kwargs = {
        "finetuned_from": model.config._name_or_path,
        "tasks": "question-answering",
        "dataset": "squad",
        "tags":["question-answering", "nlp"]
    }


    trainer.push_to_hub(commit_message = "model tuned", **kwargs)