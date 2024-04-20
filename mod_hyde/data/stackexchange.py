import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
import os
from bs4 import BeautifulSoup

def get_text_html(html_text):
    soup = BeautifulSoup(html_text, 'html.parser')

    paragraphs = soup.find_all('p')

    # Extract the text from each <p> tag
    paragraph_texts = [p.get_text() for p in paragraphs]

    # Print the extracted text
    answer_text = "".join([text for text in paragraph_texts])
    return answer_text

stack_exchange_data = load_dataset("HuggingFaceH4/stack-exchange-preferences",split="train",streaming=True)
columns = ['date','qid','question','answer_text','selected_answer_text','stack_exchange']

def is_file_empty(file_path):
    return os.path.getsize(file_path) == 0

if not os.path.exists(f"stackexchange_data.csv"):
    df = pd.DataFrame(columns=columns,index=None)
    df.to_csv(f"stackexchange_data.csv",index=False,header=True)
else:
    df = pd.read_csv(f"stackexchange_data.csv",index_col=False)

# if not os.path.exists(f"done_stackexchange.txt"):
#     os.mknod(f"done_stackexchange.txt")

# stack_exchanges= ['ai','bitcoin','anime',"datascience","astronomy"]
# qids = []
# dates = []
# question_list = []
# answer_text_list = []
# selected_answer_text_list = []
# stack_exchange_list = []
total = 50_000
progress_bar = tqdm(total=total)
curr_idx = 0
while True:
    if total == curr_idx: break
    sd = next(iter(stack_exchange_data.shuffle()))
    df_dict = {}
    best_answer_pm_score_list = [s['pm_score'] for s in sd['answers']]
    if len(list(set(best_answer_pm_score_list))) < len(best_answer_pm_score_list): 
        progress_bar.update(1)
        continue
    if "meta" in sd['metadata'][0]:
        progress_bar.update(1)
        continue

    # qids.append(sd['qid'])
    df_dict['date'] = [sd['date']]
    df_dict['qid'] = [sd['qid']]
    # dates.append(sd['date'])
    link = sd['metadata'][1]
    # stack_exchange_list.append(link[link.find("//"):link.find(".")][2:])
    df_dict['question'] = [get_text_html(sd['question'])]
    answer_text = ""
    best_answer = ""
    best_answer_pm_score = 0
    for idx,ans in enumerate(sd['answers']):
        curr_answer = get_text_html(ans['text'])
        answer_text += curr_answer + "\n\n"
        if ans['pm_score'] > best_answer_pm_score:
            best_answer = curr_answer
    # answer_text_list.append(answer_text)
    df_dict['answer_text'] = [answer_text]
    df_dict['selected_answer_text'] = [best_answer]
    df_dict['stack_exchange'] = [link[link.find("//"):link.find(".")][2:]]
    progress_bar.update(1)
    curr_df = pd.DataFrame(df_dict)
    # print(df_dict)
    curr_df.to_csv(f"stackexchange_data.csv", mode='a',index=False,header=False)
    curr_idx += 1
