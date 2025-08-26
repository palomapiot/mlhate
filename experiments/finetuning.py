import os
import wandb
import json
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorWithPadding,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from huggingface_hub import login
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template

hf_token = "hf_token"
login(hf_token)

## Data preparation
print('Load Data', flush=True)
df = pd.read_csv('data/MetaHate<lang>_train.tsv', sep='\t')
df = df[['text', 'label']]

def to_bool(value):
    return bool(int(value))


df["label"] = df["label"].apply(to_bool)

## Load model
print('Model', flush=True)
max_seq_length = 4096
dtype = None
load_in_4bit = True
seed = 47
base_model_id = 'unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit' # unsloth/Mistral-Nemo-Instruct-2407-bnb-4bit
new_model = "hate-<lang>-<model>"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=base_model_id,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
    token=hf_token
)

model = FastLanguageModel.get_peft_model(
    model=model,
    r=8,
    target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
    lora_alpha=16,
    lora_dropout=0,  # 0 is optimized with unsloth
    bias='none',  # 'none' is optimized with unsloth
    use_gradient_checkpointing='unsloth',  # use 'unsloth' to reduce vram usage
    random_state=seed,
    max_seq_length=max_seq_length,
    use_rslora=False,  # rank stabilized lora
)


def row_to_list(row):
    return [
        {"from": "human", "value": row["text"]},
        {"from": "gpt", "value": str(row["label"])}
    ]


df["conversations"] = df.apply(row_to_list, axis=1)
df["conversations"] = df["conversations"].apply(json.dumps)
df_conversations = df[["conversations"]]
dataset = Dataset.from_pandas(df_conversations)

tokenizer = get_chat_template(
    tokenizer,
    chat_template="<llama-3, mistral>",
    mapping={"role": "from", "content": "value", "user": "human", "assistant": "gpt"},
)


def formatting_prompts_func(examples):
    conversations = [json.loads(c) for c in examples["conversations"]]
    texts = [
        tokenizer.apply_chat_template(c, tokenize=False, add_generation_prompt=False)
        for c in conversations
    ]
    return {"text": texts}


dataset = dataset.map(formatting_prompts_func, batched=True)
print(dataset[0], flush=True)

### Track training metrics ###
wandb.login(key='token')
wandb_project = ("hate-<lang>-<model>")
if len(wandb_project) > 0:
    os.environ["WANDB_PROJECT"] = wandb_project

print('Set training arguments', flush=True)
training_arguments = TrainingArguments(
    output_dir="/outputs",
    num_train_epochs=2,
    per_device_train_batch_size=12,
    gradient_accumulation_steps=4,
    optim="adamw_8bit",
    save_steps=100,
    logging_steps=10,
    learning_rate=2e-5,
    weight_decay=0.01,
    fp16=not is_bfloat16_supported(),
    bf16=is_bfloat16_supported(),
    warmup_ratio=0.03,
    lr_scheduler_type="linear",
    seed=seed,
    report_to="wandb"
)

trainer = SFTTrainer(
    model=model,
    args=training_arguments,
    train_dataset=dataset,
    dataset_text_field="text",
    tokenizer=tokenizer,
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=False
)

## Train the model
print('Train!', flush=True)
trainer_stats = trainer.train()

## Save model
model.save_pretrained("model/<model>-hate-<lang>")
tokenizer.save_pretrained("model/<model>-hate-<lang>")
wandb.finish()
model.config.use_cache = True
print('saved!', flush=True)
