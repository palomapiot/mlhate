import re

import json_repair, random
import pandas as pd
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from tqdm import tqdm

print('Load data', flush=True)
df = pd.read_csv('data/MetaHate<lang>_test.tsv', sep='\t')

instruction = "Determine whether a social media message is hateful or not. Respond with either a True for hate speech or False for neutral. Do not include nothing else than True or False."
sentences = df['text'].to_list()

messages = []

for sentence in sentences:
    messages.append([{"from": "human", "value": instruction + " <Message>" + sentence + "</Message>"}])  # zero-shot


print('Load saved model', flush=True)
max_seq_length = 4096
dtype = None
load_in_4bit = True

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit", # unsloth/Mistral-Nemo-Instruct-2407-bnb-4bit
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
    token="hf_token"
)

tokenizer = get_chat_template(
    tokenizer,
    chat_template="<llama-3, mistral>",
    mapping={"role": "from", "content": "value", "user": "human", "assistant": "gpt"},
)

FastLanguageModel.for_inference(model)

generated = []
for message in tqdm(messages):
    inputs = tokenizer.apply_chat_template(
        message,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to("cuda")

    outputs = model.generate(
        input_ids=inputs,
        max_new_tokens=256,
        use_cache=True,
        temperature=0.01,
        top_p=0.1,
        top_k=5
    )
    result = tokenizer.batch_decode(outputs)
    pattern = r"<\|start_header_id\|>assistant<\|end_header_id\|>(.*?)<\|eot_id\|>"
    #pattern = r"\[\/INST\](.*?)<\/s>"
    matches = re.findall(pattern, result[0], re.DOTALL)
    output = matches[0].replace('\r', '').replace('\n', '')
    generated.append(output)

df['prediction'] = generated
df.to_csv('data/<model>-zero-shot-<lang>.tsv', sep='\t', index=False)
print('saved!', flush=True)
