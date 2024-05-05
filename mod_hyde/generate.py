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
# model = Qwen2ForCausalLM.from_pretrained(pretrained_model_name_or_path="qwensmall-full_parameter/checkpoint-5964")
# tokenizer = Qwen2Tokenizer.from_pretrained("Qwen/Qwen1.5-0.5B",)
# pipe = pipeline("text-generation", model=model,tokenizer=tokenizer)
# question = "Mitochondria play a role in remodelling lace plant leaves during programmed cell death"
# out = pipe(question,max_new_tokens=256,do_sample=True,min_new_tokens=10,top_p=0.95)
# print(out[0]['generated_text'])

peft_model_id = "Mod-HyDE/mod_hyde/qwensmall-lora--8--wiki/checkpoint-831"
config = PeftConfig.from_pretrained(peft_model_id)
# load the base LM
tokenizer = Qwen2Tokenizer.from_pretrained("Qwen/Qwen1.5-0.5B")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
# merge LNTuning weights into the base LM
model = PeftModel.from_pretrained(model, peft_model_id)

model.to(device)
model.eval()
i = 4
inputs = tokenizer('Does mitchondria ', return_tensors="pt")

with torch.no_grad():
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model.generate(
        input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], max_new_tokens=10, eos_token_id=3
    )
    print(outputs)
    print(tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True))