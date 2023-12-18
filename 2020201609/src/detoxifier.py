from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

print("loading...", end=" ")

model_path = "trained_models/mbart_40000_ENG/checkpoint-3000"
model = MBartForConditionalGeneration.from_pretrained(model_path)
tokenizer = MBart50TokenizerFast.from_pretrained(model_path)

print("done.")
print("input your TOXIC sentence: ")
input_text = input()

model_input = tokenizer([input_text], return_tensors="pt", padding=True)['input_ids']
model_output = model.generate(model_input,      
                              num_return_sequences=1,
                              do_sample=False,
                              temperature=1.0,
                              repetition_penalty=10.0,
                              max_length=model_input.shape[1] + 10,
                              min_length=int(0.5 * (model_input.shape[1] + 10)),
                              num_beams=5
                              )

# print(model_output)
# output_text = tokenizer.decode(model_output, skip_special_tokens=True)
output_text = [tokenizer.decode(out, skip_special_tokens=True) for out in model_output]
print(output_text[0])