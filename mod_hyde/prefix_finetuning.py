from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, default_data_collator, get_linear_schedule_with_warmup,get_cosine_schedule_with_warmup
from peft import get_peft_config, get_peft_model, get_peft_model_state_dict, PrefixTuningConfig, TaskType
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import os
from typing import List
import pandas as pd
from datasets import Dataset
from transformers import AutoModelForCausalLM

os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def get_hf_dataset(all_text_list:List[str]):
    df = pd.DataFrame(all_text_list,columns=["text"])
    hf_dataset = Dataset.from_pandas(df)
    return hf_dataset
with open("data/pubmed.txt","r") as f:
  all_text_list = f.readlines()
hf_dataset = get_hf_dataset(all_text_list)
device = "cuda"
model_name_or_path = "Qwen/Qwen1.5-0.5B"
tokenizer_name_or_path = "Qwen/Qwen1.5-0.5B"

text_column = "text"
max_length = 1024
lr = 2e-5
num_epochs = 3
batch_size = 4
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
num_query = 10
def preprocess_function(examples):
    inputs = []
    targets = []
    for ex in examples[text_column]:
      words_list = ex.split(" ")
      input = " ".join(words_list[:num_query])
      inputs.append(input)
      target = " ".join(words_list[num_query:])
      targets.append(target)
    model_inputs = tokenizer(inputs, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt")
    labels = tokenizer(targets, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt")
    labels = labels["input_ids"]
    labels[labels == tokenizer.pad_token_id] = -100
    model_inputs["labels"] = labels
    return model_inputs

processed_datasets = hf_dataset.map(
    preprocess_function,
    batched=True,
    num_proc=1,
    remove_columns=['text'],
    load_from_cache_file=False,
    desc="Running tokenizer on dataset",
)

train_dataloader = DataLoader(
    processed_datasets, shuffle=True, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True
)

peft_config = PrefixTuningConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, num_virtual_tokens=20)

model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
cosine_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=(len(train_dataloader) * num_epochs),
)
device = "cuda:0" if torch.cuda.is_available() else 'cpu'

model = model.to(device)

for epoch in range(num_epochs):
    # model.train()
    total_loss = 0
    for step, batch in enumerate(tqdm(train_dataloader)):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        total_loss += loss.detach().float()
        loss.backward()
        optimizer.step()
        cosine_scheduler.step()
        optimizer.zero_grad()
    torch.save(model.state_dict(), f"Qwen_Prefix_{epoch}")
        

    # model.eval()
    # eval_loss = 0
    # eval_preds = []
    # for step, batch in enumerate(tqdm(eval_dataloader)):
    #     batch = {k: v.to(device) for k, v in batch.items()}
    #     with torch.no_grad():
    #         outputs = model(**batch)
    #     loss = outputs.loss
    #     eval_loss += loss.detach().float()
    #     eval_preds.extend(
    #         tokenizer.batch_decode(torch.argmax(outputs.logits, -1).detach().cpu().numpy(), skip_special_tokens=True)
    #     )

    # eval_epoch_loss = eval_loss / len(eval_dataloader)
    # eval_ppl = torch.exp(eval_epoch_loss)
    train_epoch_loss = total_loss / len(train_dataloader)
    train_ppl = torch.exp(train_epoch_loss)
    print(f"{epoch=}: {train_ppl=} {train_epoch_loss=}")


from peft import PeftModel, PeftConfig

inputs = tokenizer(
    "Do mitochondria play a role in remodelling lace plant leaves during programmed cell death?",
    return_tensors="pt",
)


with torch.no_grad():
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model.generate(input_ids=inputs["input_ids"], max_new_tokens=10)
    print(tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True))