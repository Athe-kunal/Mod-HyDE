question_words = ['What', 'Who', 'Whom', 'Whose', 'Which', 'Where', 'When', 'Why', 'How']
auxiliary_verbs = ['Do', 'Does', 'Did', 'Is', 'Are', 'Was', 'Were', 'Can', 'Could', 'Will', 'Would']
all_questions_words = []
for qw in question_words:
    all_questions_words.append(qw)
for qw in auxiliary_verbs:
    all_questions_words.append(qw)
# print(s.replace('?','').replace())
with open('data/wikipedia_qs.txt','r') as f:
    questions = f.readlines()
formatted_questions = []
for ques in questions:
    formatted_ques = ques.replace('?','')
    filtered_words = [word for word in formatted_ques.split(" ") if word not in all_questions_words]
    formatted_questions.append(" ".join(filtered_words))

with open('data/wikipedia_formatted_qs.txt','w') as f1:
    for qs in formatted_questions:
        f1.write(f"{qs}")
with open('data/pubmed_qs.txt','r') as f:
    questions = f.readlines()
formatted_questions = []
for ques in questions:
    formatted_ques = ques.replace('?','')
    filtered_words = [word for word in formatted_ques.split(" ") if word not in all_questions_words]
    print(ques)
    formatted_questions.append(" ".join(filtered_words))

with open('data/pubmed_formatted_qs.txt','w') as f1:
    for qs in formatted_questions:
        f1.write(f"{qs}")
