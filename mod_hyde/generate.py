from unsloth import FastLanguageModel
from transformers import pipeline,AutoModelForCausalLM, AutoTokenizer, Qwen2ForCausalLM, Qwen2Tokenizer
from peft import PeftConfig, PeftModel
import torch
# model, tokenizer = FastLanguageModel.from_pretrained(
#             model_name = "gemma-lora-8/checkpoint-489", # Supports Llama, Mistral - replace this!
#             max_seq_length = 1024,
#             dtype = None,
#             load_in_4bit = False,
#         )
# config = PeftConfig.from_pretrained("tinyllama-lora-1024/checkpoint-1000")
# model = AutoModelForCausalLM.from_pretrained(
#     config.base_model_name_or_path,
# )
# tokenizer =
# state_dict = torch.load("Qwen_Prefix/Qwen_Prefix_2") 
# print(state_dict)
# model = Qwen2ForCausalLM.from_pretrained(pretrained_model_name_or_path="Qwen/Qwen1.5-0.5B",state_dict=state_dict)
model = Qwen2ForCausalLM.from_pretrained(pretrained_model_name_or_path="qwensmall-full_parameter/checkpoint-5964")
tokenizer = Qwen2Tokenizer.from_pretrained("Qwen/Qwen1.5-0.5B",)
pipe = pipeline("text-generation", model=model,tokenizer=tokenizer)
question = "Mitochondria play a role in remodelling lace plant leaves during programmed cell death"
out = pipe(question,max_new_tokens=256,do_sample=True,min_new_tokens=10,top_p=0.95)
print(out[0]['generated_text'])