import random

def format_question_generation(topic): # This is the prompt for question generation
    prompt = f"Consider the following topic: {topic}.\nPropose only one question to query information about which you lack knowledge. The question should be short and concise."
    return prompt

def format_question_w_document(topic,document):
    prompt = f"{document}\n\nInstruction: propose only one question regarding \"{topic}\".\nQuestion: "
    return prompt

def format_answer(question,document=''):
    if document != '':
        prompt = f"Information: {document}\n\nUse the information provided to answer the following question.\nQuestion: {question}\nAnswer: "
    else:
        prompt = f"Question: {question}\nAnswer: "
    return prompt

self_reflection_prompt = "Question:\n{instruction}\nProposed Answer:\n{answer}\nIs the proposed answer: (A) Correct (B) Incorrect (C) I am not sure. Choose one letter from among choices A through C.\nSelected choice:" ## This is the prompt for self reflection

self_answer_prompt = "Given a question and a document, provide an answer by using information from the document.\nNote that the document only serves to provide helpful information and should not be explicitly mentioned. Answer the question concisely without elaboration.\nQuestion:\n{question}\nDocument:\n{document}\nAnswer:" ## This is for the model to self generate answer with document

def QA_format(instr,choices):
    return f'Q: {instr}\nChoices:\n' + '\n'.join([f'{chr(i+97).upper()}: {c}' for i,c in enumerate(choices)]) + '\nAnswer:'

def QA_format_answer(answer,choices): # answer is the index of the correct choice
    return f'{chr(answer+97).upper()}: {choices[answer]}'

question_generation_examples = [ ## Based on TriviaQA wiki validation set
    {'topic':'Niamey',
     'question':"Of which African country is Niamey the capital?",
     "document":"Niamey is the capital and largest city of the West African country Niger. Niamey lies on the Niger River, primarily situated on the east bank. It is an administrative, cultural and economic centre. Niamey's population, which was estimated at 774,235 in 2006, is now projected to be much higher.",
     "answer":"The country is Niger."},
    
    {'topic':'Stagecoach (1939 film)',
     'question':"Who directed the classic 30s western Stagecoach?",
     "document":"Stagecoach is a 1939 American Western film directed by John Ford, starring Claire Trevor and John Wayne in his breakthrough role. The screenplay, written by Dudley Nichols, is an adaptation of \"The Stage to Lordsburg\", a 1937 short story by Ernest Haycox.",
     "answer":"John Ford directed Stagecoach."},
    
    {'topic':'If I Were a Rich Man (song)',
     'question':"If I Were A Rich Man Was a big hit from which stage show?",
     "document":"\"\"If I Were a Rich Man\" is a popular song from the 1964 musical Fiddler on the Roof. It was written by Sheldon Harnick and Jerry Bock. The song is performed by Tevye, the main character in the musical, and reflects his dreams of glory. \n\nThe title is inspired by a 1902 monologue by Sholem Aleichem in Yiddish, Ven ikh bin a Rothschild (If I were a Rothschild), a reference to the wealth of the Rothschild family, although the content is quite different.",
     "answer":"The stage show is Fiddler on the Roof."},
    
    {'topic':'Gerald Ford',
     'question':"What was President Gerald Ford's middle name?",
     "document":"Gerald Rudolph Ford Jr. (born Leslie Lynch King Jr.; July 14, 1913 – December 26, 2006) was an American politician who served as the 38th President of the United States from 1974 to 1977. Prior to this he was the 40th Vice President of the United States, serving from 1973 until President Richard Nixon's resignation in 1974.",
     "answer":"President Gerald Ford middle name is Rudolph."},
    
    {'topic':'River Phoenix',
     'question':"River Phoenix died during the making of which movie?",
     "document":"River Jude Phoenix (born River Jude Bottom; August 23, 1970 – October 31, 1993 ) was an American actor, musician, and activist.\n\nOn October 31, 1993, Phoenix collapsed and died of drug-related Heart Failure on the sidewalk outside the West Hollywood nightclub The Viper Room at the age of 23. At the time of his death, Phoenix had been in the middle of filming Dark Blood (1993).",
     "answer":"River Phoenix died making Dark Blood."},
    
    {'topic':'Kathleen Ferrier',
     'question':"What claimed the life of singer Kathleen Ferrier?",
     "document":"Kathleen Mary Ferrier, CBE (22 April 1912 - 8 October 1953) was an English contralto singer who achieved an international reputation as a stage, concert and recording artist, with a repertoire extending from folksong and popular ballads to the classical works of Bach, Brahms, Mahler and Elgar. Her death from cancer, at the height of her fame, was a shock to the musical world and particularly to the general public, which was kept in ignorance of the nature of her illness until after her death.",
     "answer":"Kathleen Ferrier died of cancer."},
    
    {'topic':'Lauren Bacall',
     'question':"Which actress was voted Miss Greenwich Village in 1942?",
     "document":"Lauren Bacall (, born Betty Joan Perske; September 16, 1924 – August 12, 2014) was an American actress known for her distinctive voice and sultry looks. She was named the 20th greatest female star of Classic Hollywood cinema by the American Film Institute, and received an Academy Honorary Award from the Academy of Motion Picture Arts and Sciences in 2009, \"in recognition of her central place in the Golden Age of motion pictures.\n\nShe made her acting debut on Broadway in 1942, at age 17, as a walk-on in Johnny 2 X 4. By then, she lived with her mother on Bank Street, Greenwich Village, and in 1942 she was crowned Miss Greenwich Village.",
     "answer":"Lauren Bacall was voted Miss Greenwich Village in 1942."},
    
    {'topic':'Evening Class (novel)',
     'question':"Who wrote the novel Evening Class in 1996?",
     "document":"Evening Class is a 1996 novel by the Irish author Maeve Binchy. It was adapted as the award-winning film Italian for Beginners (2000) by writer-director Lone Scherfig, who failed to formally acknowledge the source, although at the very end of the closing credits is the line 'with thanks to Maeve Binchy'.\n\nPlot: A story of many Irish men and women from various backgrounds and how a teacher, Nora O'Donoghue (known as \"Signora\"), and an Italian evening class changes their lives over the course of a year. Each chapter deals with the life story of one or more students in the class. In a Dickensian way, they bump into each other and are affected by the decisions of those around them.",
     "answer":"Maeve Binchy wrote Evening Class."},
]