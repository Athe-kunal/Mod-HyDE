from datasets import load_dataset
from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm

stack_exchange_data = load_dataset("HuggingFaceH4/stack-exchange-preferences",split="train",streaming=True)
stackexchange_subset_data = stack_exchange_data.take(1000_000)
stack_exchange_data_list = list(stackexchange_subset_data)

def get_text_html(html_text):
    soup = BeautifulSoup(html_text, 'html.parser')

    all_text = soup.get_text()
    return all_text

columns = ['date','qid','question','answer_text','selected_answer_text','stack_exchange']
df = pd.DataFrame(columns=columns)

qids = []
dates = []
question_list = []
answer_text_list = []
selected_answer_text_list = []
stack_exchange_list = []
for sd in tqdm(stack_exchange_data_list):
    best_answer_pm_score_list = [s['pm_score'] for s in sd['answers']]
    if len(list(set(best_answer_pm_score_list))) < len(best_answer_pm_score_list): continue
    if "meta" in sd['metadata'][0]:continue

    qids.append(sd['qid'])
    dates.append(sd['date'])
    link = sd['metadata'][1]
    stack_exchange_list.append(link[link.find("//"):link.find(".")][2:])
    question_list.append(get_text_html(sd['question']))
    answer_text = ""
    best_answer = ""
    best_answer_pm_score = 0
    for idx,ans in enumerate(sd['answers']):
        answer_text += get_text_html(ans['text']) + "\n\n\n\n"
        if ans['pm_score'] > best_answer_pm_score:
            best_answer = answer_text.split("\n\n\n\n")[0]

    answer_text_list.append(answer_text)
    selected_answer_text_list.append(best_answer)

df['qid'] = qids
df['date'] = dates
df['question'] = question_list
df['answer_text'] = answer_text_list
df['selected_answer_text'] = selected_answer_text_list
df['stack_exchange'] = stack_exchange_list

df.to_csv("StackExchangeData.csv",index=False)