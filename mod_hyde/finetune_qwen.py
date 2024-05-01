from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List
from datasets import Dataset
import pandas as pd
from config import *
import argparse
# from unsloth import FastLanguageModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Qwen2Tokenizer, Qwen2ForCausalLM
from transformers import Trainer, TrainingArguments
from peft import get_peft_config, get_peft_model, get_peft_model_state_dict, LoraConfig, TaskType
import json

parser = argparse.ArgumentParser(description='Finetune the model')
parser.add_argument("-m","--model", type=str,help="Name of the model for finetuning on the raw texts",default="qwensmall")
parser.add_argument("-ds","--dataset", type=str,default='pubmed',help="Dataset name")
parser.add_argument("-bls","--block_size", type=int, help="Block size for the finetuning",default=1024)
parser.add_argument("-ft","--finetuning_type", type=str, help="Finetuning type out of lora, qlora and full-parameter",default="full_parameter")
parser.add_argument("-cs","--chunk_size",type=int, help="Chunksize of the raw texts",default=1750)
parser.add_argument("-co","--chunk_overlap",type=int, help="Chunksize overlap of the raw texts",default=100)
parser.add_argument("-tbs","--tokenizer_batch_size",type=int, help="Batch size of tokenization",default=2000)
parser.add_argument("-np","--num_proc",type=int, help="number of processes",default=4)
parser.add_argument("-bs","--batch_size",type=int, help="Batch size of finetuning",default=4)
parser.add_argument("-nte","--num_train_epochs",type=int, help="Number of training epochs",default=1)
parser.add_argument("-r","--rank",type=int, help="Number of ranks",default=16)
parser.add_argument("-a","--alpha",type=int, help="LoRA alpha",default=16)
parser.add_argument("-re","--rerun",type=bool, help="Rerun from checkpoint",default=False)
parser.add_argument("-lr","--learning_rate",type=float, help="Learning rate",default=2e-5)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
args = parser.parse_args()

def split_text_langchain(text):
    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        # separators = [". "],
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

# def group_texts(examples):
#     # Concatenate all texts.
#     concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
#     total_length = len(concatenated_examples[list(examples.keys())[0]])
#     # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
#         # customize this part to your needs.
#     total_length = (total_length // args.block_size) * args.block_size
#     # Split by chunks of max_len.
#     result = {
#         k: [t[i : i + args.block_size] for i in range(0, total_length, args.block_size)]
#         for k, t in concatenated_examples.items()
#     }
#     result["labels"] = result["input_ids"].copy()
#     return result


def tokenize_dataset(all_text_list:List[str],tokenizer):
    hf_dataset = get_hf_dataset(all_text_list)
    def tokenize_function(examples):
        return tokenizer(examples["text"])
    def group_texts(examples):
        # Concatenate all texts.
        examples = tokenize_function(examples)
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
    # tokenized_datasets = hf_dataset.map(tokenize_function, batched=True,  remove_columns=["text"])
    lm_datasets = hf_dataset.map(
        group_texts,
        batched=True,
        batch_size=args.tokenizer_batch_size,
        remove_columns=["text"]
    )
    return lm_datasets


def main(all_text_list):
    if args.finetuning_type != "full_parameter" or args.finetuning_type != "lora":
        load_in_4bit = False
    else:
        load_in_4bit = True

    if args.model == "qwenbig" or args.model == "qwensmall":
        if args.model == "qwenbig":
            qwen_model = "Qwen/Qwen1.5-1.8B"
        elif args.model == "qwensmall":
            qwen_model = "Qwen/Qwen1.5-0.5B"
        tokenizer = Qwen2Tokenizer.from_pretrained(qwen_model,trust_remote_code=True)
        model = Qwen2ForCausalLM.from_pretrained(qwen_model,trust_remote_code=True,attn_implementation="flash_attention_2")
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
        output_dir=f"{args.model}-{args.finetuning_type}--{args.rank}--{args.dataset}",
        num_train_epochs=args.num_train_epochs,
        torch_compile=True,
        per_device_train_batch_size=args.batch_size,
        fp16 = not torch.cuda.is_bf16_supported(),
        learning_rate=args.learning_rate,
        bf16 = torch.cuda.is_bf16_supported(),
        weight_decay=0.01,
        lr_scheduler_type= 'cosine',
        optim = "paged_adamw_32bit",
        seed = 42,
        # tf32=True,
        gradient_accumulation_steps = 1,save_strategy='epoch')
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_datasets
    )
    if args.rerun:
        trainer.train(resume_from_checkpoint = True)
    else:
        trainer.train()
    # if args.finetuning_type != "full_parameter":

    #     trainer.save_model(f"{args.model}-{args.finetuning_type}")
    
    # elif args.finetuning_type != "lora" and args.finetuning_type != "qlora":
    #     trainer.save_model(f"{args.model}-{args.finetuning_type}-{args.block_size}-{args.num_train_epochs}")


if __name__ == "__main__":
    # import json
    if args.dataset == 'pubmed':
        with open("data/pubmed.txt","r") as f:
            all_text_list = f.readlines()
    elif args.dataset == 'wiki':
        with open("data/wikipedia_filtered.txt","r") as f:
            wiki_data = f.read()
        wiki_data_list = wiki_data.split("-----------")[:-1]
        wiki_data_list = [at.strip() for at in wiki_data_list]
        all_text_list = []

        for text in wiki_data_list:
            splitted_text = split_text_langchain(text)
            all_text_list.extend(splitted_text)

    main(all_text_list)
