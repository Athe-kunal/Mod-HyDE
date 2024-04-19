from langchain_community.document_loaders import WikipediaLoader
import wikipediaapi
import backoff
import requests
from datasets import load_dataset
hotpot_qa = load_dataset("hotpot_qa","distractor")
val_subset = hotpot_qa['validation']['context']
titles = []
for sd in val_subset:
    titles.extend(sd['title'])

wikipedia_module = wikipediaapi.Wikipedia("HyDE-Project (athekunal@gmail.com)", "en")

@backoff.on_exception(backoff.expo,requests.exceptions.Timeout,max_tries=5,max_time=30)
def get_wikipedia(title):
    page_py = wikipedia_module.page(title)
    title_text = ""
    for st in page_py.sections:
        if st.title.lower() == 'references' or st.title.lower() == 'external links': continue
        title_text += st.text + "\n"
    return title_text

from tqdm import tqdm
import os
import pandas as pd

def is_file_empty(file_path):
    return os.path.getsize(file_path) == 0

if not os.path.exists(f"wikipedia_data_val.csv"):
    df = pd.DataFrame(columns=["title","text"],index=None)
    df.to_csv(f"wikipedia_data_val.csv",index=False,header=True)
else:
    df = pd.read_csv(f"wikipedia_data_val.csv",index_col=False)

if not os.path.exists(f"done_wikipedia_val.txt"):
    os.mknod(f"done_wikipedia_val.txt")

for title in tqdm(titles[57612:]):
    # docs = WikipediaLoader(
    #         query=title, load_max_docs=1, doc_content_chars_max=1
    #     ).load()
    # wikipedia_title = docs[0].metadata["title"]
    if not is_file_empty(f"done_wikipedia_val.txt"):
        with open(f"done_wikipedia_val.txt","r") as f:
            done_titles = [x for x in f.read().splitlines()]
        # print(done_pids)
        if title in done_titles: 
            continue
    title_text = get_wikipedia(title)
    df_dict = {
        "title":[title],
        "text":[title_text]
    }
    df = pd.DataFrame(df_dict)
    
    df.to_csv(f"wikipedia_data_val.csv", mode='a',index=False,header=False)
    with open(f"done_wikipedia_val.txt", "a") as f:
        f.write(f"{title}\n")