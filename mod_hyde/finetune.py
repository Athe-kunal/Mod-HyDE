from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List
from datasets import Dataset
import pandas as pd
from config import *
import argparse
from unsloth import FastLanguageModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import Trainer, TrainingArguments
from peft import get_peft_config, get_peft_model, get_peft_model_state_dict, LoraConfig, TaskType

parser = argparse.ArgumentParser(description='Finetune the model')
parser.add_argument("-m","--model", type=str,help="Name of the model for finetuning on the raw texts")
parser.add_argument("-bls","--block_size", type=int, help="Block size for the finetuning")
parser.add_argument("-ft","--finetuning_type", type=str, help="Finetuning type out of lora, qlora and full-parameter")
parser.add_argument("-cs","--chunk_size",type=int, help="Chunksize of the raw texts",default=1750)
parser.add_argument("-co","--chunk_overlap",type=int, help="Chunksize overlap of the raw texts",default=100)
parser.add_argument("-tbs","--tokenizer_batch_size",type=int, help="Batch size of tokenization",default=2000)
parser.add_argument("-np","--num_proc",type=int, help="number of processes",default=4)
parser.add_argument("-bs","--batch_size",type=int, help="Batch size of finetuning",default=4)
parser.add_argument("-nte","--num_train_epochs",type=int, help="Number of training epochs",default=1)
parser.add_argument("-r","--rank",type=int, help="Number of ranks",default=16)
parser.add_argument("-a","--alpha",type=int, help="LoRA alpha",default=16)
parser.add_argument("-lr","--learning_rate",type=float, help="Learning rate",default=2e-5)

args = parser.parse_args()

def split_text(text):
    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        separators = [". "],
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        length_function=len,
        is_separator_regex=False)

    splitted_text = text_splitter.split_text(text)
    return splitted_text

def get_hf_dataset(all_text_list:List[str]):
    df = pd.DataFrame(all_text_list,columns=["text"])
    hf_dataset = Dataset.from_pandas(df)
    return hf_dataset

def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
    total_length = (total_length // args.block_size) * args.block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + args.block_size] for i in range(0, total_length, args.block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


def tokenize_dataset(all_text_list:List[str],tokenizer):
    hf_dataset = get_hf_dataset(all_text_list)
    def tokenize_function(examples):
        return tokenizer(examples["text"])
    tokenized_datasets = hf_dataset.map(tokenize_function, batched=True, num_proc=args.num_proc, remove_columns=["text"])
    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        batch_size=args.tokenizer_batch_size,
        num_proc=args.num_proc,
    )
    return lm_datasets


def main(all_text_list):
    if args.model == "tinyllama":
        model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/tinyllama", # Supports Llama, Mistral - replace this!
        max_seq_length = args.block_size,
        dtype = None,
        load_in_4bit = True,
    )
        if args.finetuning_type == "qlora" or args.finetuning_type == "lora":
            model = FastLanguageModel.get_peft_model(
                model,
                r = args.rank, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
                target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                                "gate_proj", "up_proj", "down_proj",],
                lora_alpha = args.alpha,
                lora_dropout = 0, # Currently only supports dropout = 0
                bias = "none",    # Currently only supports bias = "none"
                use_gradient_checkpointing = True, # @@@ IF YOU GET OUT OF MEMORY - set to True @@@
                random_state = 3407,
                use_rslora = False,  # We support rank stabilized LoRA
                loftq_config = None, # And LoftQ
            )
    elif args.model == "phi2":
        model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/phi-2", # Supports Llama, Mistral - replace this!
        max_seq_length = args.block_size,
        dtype = None,
        load_in_4bit = True,
    )
        if args.finetuning_type == "qlora" or args.finetuning_type == "lora":
            model = FastLanguageModel.get_peft_model(
                model,
                r = args.rank, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
                target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                                "gate_proj", "up_proj", "down_proj",],
                lora_alpha = 32,
                lora_dropout = 0, # Currently only supports dropout = 0
                bias = "none",    # Currently only supports bias = "none"
                use_gradient_checkpointing = True, # @@@ IF YOU GET OUT OF MEMORY - set to True @@@
                random_state = 3407,
                use_rslora = False,  # We support rank stabilized LoRA
                loftq_config = None, # And LoftQ
            )
    elif args.model == "gemma":
        model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/gemma-2b", # Supports Llama, Mistral - replace this!
        max_seq_length = args.block_size,
        dtype = None,
        load_in_4bit = True,
    )
        if args.finetuning_type == "qlora" or args.finetuning_type == "lora":
            model = FastLanguageModel.get_peft_model(
                model,
                r = args.rank, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
                target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                                "gate_proj", "up_proj", "down_proj",],
                lora_alpha = args.alpha,
                lora_dropout = 0, # Currently only supports dropout = 0
                bias = "none",    # Currently only supports bias = "none"
                use_gradient_checkpointing = True, # @@@ IF YOU GET OUT OF MEMORY - set to True @@@
                random_state = 3407,
                use_rslora = False,  # We support rank stabilized LoRA
                loftq_config = None, # And LoftQ
            )

    elif args.model == "qwenbig" or args.model == "qwensmall":
        if args.model == "qwenbig":
            qwen_model = "Qwen/Qwen1.5-1.8B"
        elif args.model == "qwensmall":
            qwen_model = "Qwen/Qwen1.5-0.5B"
        tokenizer = AutoTokenizer.from_pretrained(qwen_model)
        model = AutoModelForCausalLM.from_pretrained(qwen_model)
        if args.finetuning_type == "qlora" or args.finetuning_type == "lora":
            peft_config = LoraConfig(inference_mode=False, target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                                "gate_proj", "up_proj", "down_proj"],r=args.rank, lora_alpha=args.alpha, lora_dropout=0.0)
            model = get_peft_model(model, peft_config)

    elif args.model == "gpt2":
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        model = AutoModelForCausalLM.from_pretrained("gpt2")
        # if args.finetuning_type == "qlora" or args.finetuning_type == "lora":

    lm_datasets = tokenize_dataset(all_text_list,tokenizer)
    
    training_args = TrainingArguments(
        output_dir=f"{args.model}-{args.finetuning_type}-{args.block_size}",
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.batch_size,
        fp16 = not torch.cuda.is_bf16_supported(),
        learning_rate=args.learning_rate,
        bf16 = torch.cuda.is_bf16_supported(),
        weight_decay=0.01,
        lr_scheduler_type= 'cosine',
        seed = 42,
        gradient_accumulation_steps = 1)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_datasets
    )
    
    trainer.train()
    if args.finetuning_type != "full_parameter":

        trainer.save_model(f"{args.model}-full-parameter-{args.block_size}")
    
    elif args.finetuning_type != "full_parameter":
        trainer.save_model(f"{args.model}-{args.finetuning_type}-{args.block_size}")



