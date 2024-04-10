from utils import tokenize_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List
from transformers import Trainer, TrainingArguments
from transformers import pipeline

def finetune_distilbert(model_name:str,all_text_list:List[str],logging_name:str):
    # model_name = "distilbert/distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    lm_datasets = tokenize_dataset(all_text_list,tokenizer)
    training_args = TrainingArguments(
        f"{model_name}-finetuned-{logging_name}",
        # evaluation_strategy = "epoch",
        learning_rate=2e-5,
        weight_decay=0.01,
        push_to_hub=False,
        num_train_epochs = 2
    )

    trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_datasets
    )

    trainer.train()

def get_mod_HyDE_answer(pipe,question):
    out = pipe(question,max_new_tokens=100,do_sample=True,min_new_tokens=10)
    return out[0]['generated_text']