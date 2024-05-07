from transformers import pipeline,AutoModelForCausalLM, AutoTokenizer, Qwen2ForCausalLM, Qwen2Tokenizer
from peft import PeftConfig, PeftModel
import torch
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-r','--rank',help="current rank")
parser.add_argument('-d','--dataset',help="current dataset")
parser.add_argument('-c','--checkpoint',help="current checkpoint")

args = parser.parse_args()

peft_model_id = f"qwensmall-lora--{args.rank}--{args.dataset}/checkpoint-{args.checkpoint}"
config = PeftConfig.from_pretrained(peft_model_id)
# load the base LM
tokenizer = Qwen2Tokenizer.from_pretrained("Qwen/Qwen1.5-0.5B",padding_side='left')
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
# merge LNTuning weights into the base LM
model = PeftModel.from_pretrained(model, peft_model_id)
model.to(device)
model.eval()
MAX_NEW_TOKENS = 96
BATCH_SIZE = 400

def generate_pubmed_wiki(path:str,ds_name:str):
    print(f"Now: {args.rank}_{args.checkpoint}")
    with open(path,'r') as f:
        questions = f.readlines()
        questions = [ques.replace("\n","") for ques in questions]
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenized_questions = tokenizer.batch_encode_plus(questions, padding=True, truncation=True, return_tensors="pt")
        tokenized_questions = {k:v.to(device) for k,v in tokenized_questions.items()}
        greedy_outputs_list = []
        nucleus_outputs_list = []
        for start in range(0,len(questions),BATCH_SIZE):
            end = min(start+BATCH_SIZE,len(questions))
            batch_outputs_input_ids = tokenized_questions['input_ids'][start:end]
            batch_outputs_attention_mask = tokenized_questions['attention_mask'][start:end]
            # if args.decoding_type == 'greedy':
            greedy_outputs = model.generate(input_ids=batch_outputs_input_ids, attention_mask=batch_outputs_attention_mask, max_new_tokens=MAX_NEW_TOKENS)
            greedy_outputs_str = tokenizer.batch_decode(greedy_outputs.detach().cpu().numpy(), skip_special_tokens=True)
            # elif args.decoding_type == 'nucleus': 
            nucleus_outputs = model.generate(input_ids=batch_outputs_input_ids, attention_mask=batch_outputs_attention_mask, max_new_tokens=MAX_NEW_TOKENS,top_p=0.95,do_sample=True,)
            nucleus_outputs_str = tokenizer.batch_decode(nucleus_outputs.detach().cpu().numpy(), skip_special_tokens=True)
            greedy_outputs_list.extend(greedy_outputs_str)
            nucleus_outputs_list.extend(nucleus_outputs_str)
            questions_list = questions[start:end]
            for ques,out in zip(questions_list,greedy_outputs_list):
                with open(f"data/answers/{ds_name}_qs_ans_{args.rank}_greedy_{args.checkpoint}.txt",'a') as f1:
                    f1.write(out.replace(ques,'').strip())
                    f1.write("\n<END>\n")
            for ques,out in zip(questions_list,nucleus_outputs_list):
                with open(f"data/answers/{ds_name}_qs_ans_{args.rank}_nucleus_{args.checkpoint}.txt",'a') as f2:
                    f2.write(out.replace(ques,'').strip())
                    f2.write("\n<END>\n")

with torch.no_grad():
    if args.dataset == 'pubmed':
        path = "data/pubmed_formatted_qs.txt"
        generate_pubmed_wiki(path,"pubmed_formatted")
        path = "data/pubmed_qs.txt"
        generate_pubmed_wiki(path,"pubmed")
    
    if args.dataset == 'wiki':
        path = "data/wikipedia_formatted_qs.txt"
        generate_pubmed_wiki(path,"wikipedia_formatted")
        path = "data/wikipedia_qs.txt"
        generate_pubmed_wiki(path,"wikipedia")
        