from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List
from datasets import Dataset
import pandas as pd
from config import *

def split_text(text):
    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        separators = [". "],
        chunk_size=1750,
        chunk_overlap=100,
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
    total_length = (total_length // BLOCK_SIZE) * BLOCK_SIZE
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + BLOCK_SIZE] for i in range(0, total_length, BLOCK_SIZE)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


def tokenize_dataset(all_text_list:List[str],tokenizer):
    hf_dataset = get_hf_dataset(all_text_list)
    def tokenize_function(examples):
        return tokenizer(examples["text"])
    tokenized_datasets = hf_dataset.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])
    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        batch_size=1000,
        num_proc=4,
    )
    return lm_datasets




