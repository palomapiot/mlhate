import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments


def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    labels = p.label_ids
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds)
    recall = recall_score(labels, preds)
    f1 = f1_score(labels, preds, average='macro')
    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}


class HSDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = self.data.iloc[index]['text']
        label = self.data.iloc[index]['label']

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids = inputs['input_ids'].flatten()
        attention_mask = inputs['attention_mask'].flatten()

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': torch.tensor(label, dtype=torch.long)
        }


def objective(path, tokenizer_name, model_name, output_dir, file_path):
    epochs = 3
    learning_rate = 5e-5
    per_device_train_batch_size = 32
    weight_decay = 0.18138151333103786
    optim = 'adamw_8bit'
    max_len = 512

    print(f'{file_path}', flush=True)
    df = pd.read_csv(file_path, sep='\t')
    df = df[['label', 'text']]
    print(df.head(), flush=True)

    tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
    train_df, eval_df = train_test_split(df, test_size=0.2, random_state=42)

    train_dataset = HSDataset(train_df, tokenizer, max_len=max_len)
    eval_dataset = HSDataset(eval_df, tokenizer, max_len=max_len)

    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

    training_args = TrainingArguments(
        output_dir=f'{path}{output_dir}',
        num_train_epochs=epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=16,
        warmup_steps=500,
        weight_decay=weight_decay,
        logging_dir=f'{path}/logs',
        logging_steps=10,
        evaluation_strategy='epoch',
        learning_rate=learning_rate,
        optim=optim
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics
    )

    trainer.train()
    trainer.save_model(f'{path}{output_dir}/trainer')
    model.save_pretrained(f'{path}{output_dir}/model')

    eval_results = trainer.evaluate(eval_dataset)
    accuracy = eval_results['eval_accuracy']
    precision = eval_results['eval_precision']
    recall = eval_results['eval_recall']
    f1 = eval_results['eval_f1']

    print(f'Accuracy: {accuracy}', flush=True)
    print(f'Precision: {precision}', flush=True)
    print(f'Recall: {recall}', flush=True)
    print(f'F1 Score: {f1}', flush=True)


def main():
    objective(
        'data/',
        'bert-base-multilingual-cased',
        'bert-base-multilingual-cased',
        'bert-<lang>',
        'data/MetaHate<lang>_train.tsv'
    )


if __name__ == "__main__":
    main() 