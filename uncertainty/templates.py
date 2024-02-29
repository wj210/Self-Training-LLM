
def format_question_generation(topic): # This is the prompt for question generation
    return f"Consider the following topics: {topic}. Propose only one question to query information about which you lack knowledge. Answer with only the proposed question concisely without elaboration."

self_reflection_prompt = "Question:\n{instruction}\nProposed Answer:\n{answer}\nIs the proposed answer: (A) Correct (B) Incorrect (C) I am not sure. Choose one letter from among choices A through C.\nSelected choice:" ## This is the prompt for self reflection

self_answer_prompt = "Given a question and a document, provide an answer by using information from the document.\nNote that the document only serves to provide helpful information and should not be explicitly mentioned. Answer the question concisely without elaboration.\nQuestion:\n{question}\nDocument:\n{document}\nAnswer:" ## This is for the model to self generate answer with document