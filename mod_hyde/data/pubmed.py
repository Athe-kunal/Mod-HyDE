from datasets import load_dataset
from tqdm import tqdm
import time
from langchain_community.utilities.pubmed import PubMedAPIWrapper
import argparse
import pandas as pd
import os

parser = argparse.ArgumentParser(description='PubMed API wrapper')

parser.add_argument('-s','--start',type=int,help="start index")
parser.add_argument('-e','--end',type=int,help="end index")
args = parser.parse_args()

start_idx = args.start
end_idx = args.end

pubmed_qa = load_dataset("qiaojin/PubMedQA","pqa_unlabeled")
all_pubmed_ids = [x['pubid'] for x in pubmed_qa['train']]

tool = PubMedAPIWrapper(email="athekunal@gmail.com",sleep_time=1)
pbar = tqdm(total=len(all_pubmed_ids[start_idx:end_idx]),desc="Pubmed Summary")

if not os.path.exists(f"done.txt"):
    os.mknod(f"done.txt")

def is_file_empty(file_path):
    return os.path.getsize(file_path) == 0

if not os.path.exists(f"pubmed_abstract_{start_idx}_{end_idx}.csv"):
    df = pd.DataFrame(columns=["url","pid","abstract"],index=None)
    df.to_csv(f"pubmed_abstract_{start_idx}_{end_idx}.csv",index=False,header=True)
else:
    df = pd.read_csv(f"pubmed_abstract_{start_idx}_{end_idx}.csv",index_col=False)

for idx,pid in enumerate(all_pubmed_ids[start_idx:end_idx]):
    if not is_file_empty(f"done_{start_idx}_{end_idx}.txt"):
        with open(f"done_{start_idx}_{end_idx}.txt","r") as f:
            done_pids = [int(x) for x in f.read().splitlines()]
        # print(done_pids)
        if pid in done_pids: 
            # print(f"Already done for {pid}")
            pbar.update(1)
            continue
    url = f"https://pubmed.ncbi.nlm.nih.gov/{pid}/"
    s = tool.load(query = f"({pid}[UID])")
    df_dict = {
        "pid":[pid],
        "url":[url],
        "abstract":[s[0]['Summary']]
    }
    # df_dict['url'].append(url)
    # df_dict['pid'].append(pid)
    # df_dict['abstract'].append(s[0]['Summary'])
    time.sleep(1)
    pbar.update(1)
    with open(f"done_{start_idx}_{end_idx}.txt","a") as f:
        f.write(f"{pid}\n")
    df = pd.DataFrame(df_dict)
    
    df.to_csv(f"pubmed_abstract_{start_idx}_{end_idx}.csv", mode='a',index=False,header=False)
