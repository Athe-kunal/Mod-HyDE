from transformers import pipeline,AutoModelForCausalLM, AutoTokenizer, Qwen2ForCausalLM, Qwen2Tokenizer
from peft import PeftConfig, PeftModel
import torch
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-r','--rank',help="current rank")
parser.add_argument('-d','--dataset',help="current dataset")
parser.add_argument('-c','--checkpoint',help="current checkpoint")
parser.add_argument('-de','--decoding_type',help='decoding type')

args = parser.parse_args()

peft_model_id = f"qwensmall-lora--{args.rank}--{args.dataset}/checkpoint-{args.checkpoint}"
config = PeftConfig.from_pretrained(peft_model_id)
# load the base LM
tokenizer = Qwen2Tokenizer.from_pretrained("Qwen/Qwen1.5-0.5B")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
# merge LNTuning weights into the base LM
model = PeftModel.from_pretrained(model, peft_model_id)

model.to(device)
model.eval()
# model.generation_config.cache_implementation = "static"
MAX_NEW_TOKENS = 96
with torch.no_grad():
    if args.dataset == 'pubmed':
        with open("data/pubmed_qs.txt") as f:
            questions = f.readlines()
            for ques in tqdm(questions):
                inputs = tokenizer(ques, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                if args.decoding_type == 'greedy':

                    outputs = model.generate(
                        input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], max_new_tokens=MAX_NEW_TOKENS
                    )
                elif args.decoding_type == 'nucleus': 
                    outputs = model.generate(
                        input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], max_new_tokens=MAX_NEW_TOKENS, top_p=0.95,do_sample=True,
                    )
                # print(outputs)
                outputs_str = tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)
                # print(outputs_str)
                with open(f"data/pubmed_qs_ans_{args.rank}_{args.decoding_type}_{args.checkpoint}.txt",'a') as f1:
                    f1.write(outputs_str[0].replace(ques,'').strip())
                    f1.write("\n<END>\n")
    if args.dataset == 'wiki':
        with open("data/wikipedia_qs.txt") as f:
            questions = f.readlines()
            for ques in tqdm(questions[:1]):
                print(ques)
                inputs = tokenizer(ques, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                if args.decoding_type == 'greedy':

                    outputs = model.generate(
                        input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], max_new_tokens=MAX_NEW_TOKENS
                    )
                elif args.decoding_type == 'nucleus': 
                    outputs = model.generate(
                        input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], max_new_tokens=MAX_NEW_TOKENS, top_p=0.95,do_sample=True,
                    )
                # print(outputs)
                outputs_str = tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)
                with open(f"data/pubmed_qs_ans_{args.rank}_{args.decoding_type}_{args.checkpoint}.txt",'a') as f1:
                    f1.write(outputs_str[0].replace(ques,'').strip())
                    f1.write("\n<END>\n")