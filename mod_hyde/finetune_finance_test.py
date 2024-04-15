import json
from utils import tokenize_dataset,split_text
from finetune import main
if __name__ == "__main__":
    with open("data/all_book_data.txt","r") as f:
        all_books_data = f.read()


    youtube_transcripts_list = ["chunked_misc_transcripts.json","chunked_transcripts_undergrad.json","chunked_transcripts_mba.json"]
    all_youtube_data = []
    for file in youtube_transcripts_list:
        with open('data/YouTube_API_Transcripts/'+file) as f:
            data = json.load(f)
        for id,data in data.items():
            for d in data:
                all_youtube_data.append(d['text'])
    books_data_splitted_text = split_text(all_books_data)
    all_data = books_data_splitted_text + all_youtube_data

    main(all_data)