import huggingface_hub

huggingface_hub.login("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")

import json

from datasets import load_dataset
id_dataset = 'Kudod/Thesis'
dataset = load_dataset(id_dataset, cache_dir=f'./dataset/{id_dataset}')

data_merge = []
for item in dataset['train']:
    data_merge.append(item)

for item in dataset['add_data']:
    data_merge.append(item)
    


check_point = "vlsp-2023-vllm/hoa-1b4"

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained(check_point, cache_dir=f'./cache/{check_point}_tokenizer')
model = AutoModelForCausalLM.from_pretrained(check_point,cache_dir=f'./cache/{check_point}_model')
from transformers import DataCollatorForLanguageModeling

tokenizer.pad_token = tokenizer.eos_token
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

from peft import get_peft_config, get_peft_model, LoraConfig, TaskType

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM , inference_mode=False, r=16, lora_alpha=32, lora_dropout=0.1,
)

model = get_peft_model(model=model, peft_config=peft_config)

model.print_trainable_parameters()

import torch
from torch.utils.data import Dataset

class MyCausalLanguageDataset(Dataset):
    def __init__(self, your_data, tokenizer):
        # Initialize the dataset with your data
        self.data = your_data
        self.tokenizer = tokenizer

    def __len__(self):
        # Return the total number of samples in the dataset
        return len(self.data)

    def __getitem__(self, idx):
        # Implement how to get a single sample from the dataset
        sample = self.data[idx]
        text = f"""{self.data[idx]['prompt']} {self.data[idx]['answer']}"""
                
        sample = self.tokenizer.encode_plus(
            text=text, 
            add_special_tokens=True,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            return_attention_mask=True,
            return_token_type_ids=True,
        )
        result = {
            'input_ids': sample['input_ids'].flatten(),
            'attention_mask': sample['attention_mask'].flatten(),
            'labels' : sample['input_ids'].flatten()
        }
        return result

custom_dataset_train = MyCausalLanguageDataset(data_merge, tokenizer)
custom_dataset_dev = MyCausalLanguageDataset(dataset['train'][28457:], tokenizer)


from transformers import AutoModelForCausalLM, TrainingArguments, Trainer

server = "kc"
date = "April17th"
name = "bloom-1b4"
logging_steps = 1000

training_args = TrainingArguments(
    output_dir=f"{name}_finetuned_{server}_{date}",
    evaluation_strategy="epoch",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    learning_rate=4e-5,
    num_train_epochs=5,
    # evaluation_strategy="steps",
    logging_strategy = "epoch",
    save_strategy = "epoch",
    
    load_best_model_at_end=True,
    weight_decay=0.01,
    push_to_hub=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=custom_dataset_train,
    eval_dataset=custom_dataset_dev,
    data_collator=data_collator,
)

trainer.train()

trainer.push_to_hub()
