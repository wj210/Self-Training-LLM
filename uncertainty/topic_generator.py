from sentence_transformers import SentenceTransformer
from typing import List
import os
import pickle
from copy import deepcopy
from tqdm import tqdm
import requests
from scorer import async_process
from datetime import datetime, timedelta
import time
from collections import Counter,defaultdict
import numpy as np
from templates import category_list
import random
from transformers import AutoTokenizer
from titles import *
# import faiss

def get_wordnet_nouns_and_embeddings(embedder: SentenceTransformer):
    # nltk.download('omw-1.4')
    from nltk.corpus import wordnet as wn
    dir_storage = "data"
    os.makedirs(dir_storage, exist_ok=True)
    wordnet_data_path = dir_storage + "/wordnet_data.pkl"
    if os.path.exists(wordnet_data_path):
        with open(wordnet_data_path, "rb") as dump_handle:
            wordnet_data = pickle.load(dump_handle)
    else:
        all_nouns = []
        for synset in wn.all_synsets("n"):
            lemma_names = [str(lemma.name()) for lemma in synset.lemmas()]
            lemma_descs = [str(synset.definition()) for lemma in synset.lemmas()]
            lemms = [n + " ### " + d for n, d in zip(lemma_names, lemma_descs)]
            all_nouns.extend(lemms)
        all_nouns = list(set(all_nouns))
        all_embeddings = embedder.encode(all_nouns, device="cpu", convert_to_numpy=True)
        wordnet_data = {"all_nouns": all_nouns, "all_embeddings": all_embeddings}
        with open(wordnet_data_path,'wb') as dump_handle:
            pickle.dump(wordnet_data, dump_handle)
    return wordnet_data

def get_topic_embedding_space(all_embeddings: List):
    topic_embedding_space = faiss.IndexFlatIP(all_embeddings.shape[-1])
    faiss.normalize_L2(all_embeddings)
    topic_embedding_space.add(all_embeddings)
    return topic_embedding_space

def get_embedding(topic_list,embedding_path,truncate=False):
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2", device="cuda")
    if truncate:
        max_length = 256
        tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L12-v2')
        topic_list = [tokenizer.decode(tokenizer.encode(t,add_special_tokens=False)[:max_length]) for t in tqdm(topic_list,total=len(topic_list),desc = 'Truncating topics')]
    all_embeddings = embedder.encode(topic_list, device="cuda", convert_to_numpy=True,batch_size=1024)
    saved_embeddings = {'all_embeddings':all_embeddings,'topics':topic_list}
    with open(embedding_path,'wb') as f:
        pickle.dump(saved_embeddings,f)
    return saved_embeddings


def get_related_topics(embedding_dict,query_topics:List[str],num_topics:int=-1):
    """
    Given an embedding dict which contains the topics and embeddings for each topic and a list of query topics to find related topics for,
    for each topic in query_topics: find num_topics related topics.
    """
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2", device="cpu")
    all_embeddings = embedding_dict['all_embeddings']
    topic_list = embedding_dict['topics']
    
    assert num_topics >0, "num_topics should be greater than 0"
    topic_embedding_space = get_topic_embedding_space(all_embeddings)
    all_selected_topics = defaultdict(list)
    embedded_query_topics = embedder.encode(query_topics, device="cpu", convert_to_numpy=True)
    for i,query_vec in tqdm(enumerate(embedded_query_topics),total = len(embedded_query_topics),desc = 'finding related topics'):
        existing_topics = sum(list(all_selected_topics.values()),[])
        non_duplicate_topics = []
        k = deepcopy(num_topics)
        num_tries = 0
        query_vec = np.array([query_vec],dtype=np.float32)
        faiss.normalize_L2(query_vec)
        while len(non_duplicate_topics) < num_topics:
            _, emb_indices = topic_embedding_space.search(query_vec, k)
            selected_topics = [topic_list[i] for i in emb_indices[0]]
            for t in selected_topics:
                if t not in set(existing_topics + non_duplicate_topics):
                    non_duplicate_topics.append(t)
                if len(non_duplicate_topics) >= num_topics:
                    break
            num_tries += 1
            k *= 2
            if num_tries > 5:
                print (f'Only found {len(non_duplicate_topics)} topics for {query_topics[i]}. Required: {num_topics}')
                break
        all_selected_topics[query_topics[i]].extend(non_duplicate_topics)
    return all_selected_topics

def get_related_topics_for_all(embedder,embedding_dict,query_topics:List[str],k:int=1):
    """
    for each query topic, get k nearest neighbours.
    keep a count of the topics that are selected.
    output the count.
    """
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2", device="cpu")
    all_embeddings = embedding_dict['all_embeddings']
    topic_list = embedding_dict['topics']
    topic_embedding_space = get_topic_embedding_space(all_embeddings)
    all_selected_topics = Counter()
    embedded_query_topics = embedder.encode(query_topics, device="cpu", convert_to_numpy=True)
    for query_vec in tqdm(embedded_query_topics,total = len(embedded_query_topics),desc = f'Finding related {k} related topics to construct test set'):
        query_vec = np.array([query_vec],dtype=np.float32)
        faiss.normalize_L2(query_vec)
        _, emb_indices = topic_embedding_space.search(query_vec, k)
        selected_topics = [topic_list[i] for i in emb_indices[0]]
        for t in selected_topics:
            all_selected_topics[t] += 1
    return all_selected_topics

def get_views_for_topic(topic_list:str):

    def get_view(topic):
        url_template = "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/all-agents/{title}/monthly/2023010100/2023120100"
        url = url_template.format(title = topic)
        response = requests.get(url,timeout = 10)
        try:
            if response.status_code == 200:
                data = response.json().get('items',[])
                if len(data) == 0:
                    return None
                else:
                    return {topic:sum([d['views'] for d in data])}
        except Exception as e:
            print (f'error for {topic}: {e}')
            return None
        
    topic_views = async_process(get_view,topic_list,workers=5,msg = 'Getting views for wikipedia articles')
    topic_views = [t for t in topic_views if t is not None]
    return topic_views

def get_top_articles(num_topics,topic_to_document,num_test_topics=0):
    views_path = 'data/embeddings/wikiviews.pkl'
    embedding_topic_path = 'data/embeddings/wiki_embeddings.pkl'
    category_topic_path = f'data/embeddings/category_topics_{int(num_topics)}.pkl'
    train_test_topics_path = f'data/embeddings/train_test_topics.pkl'
    if not os.path.exists(category_topic_path) or (os.path.exists(category_topic_path) and num_topics > int(category_topic_path.split('_')[-1].split('.')[0].strip())):
        if not os.path.exists(views_path):
            start_date = datetime(2020, 1, 1)
            end_date = datetime(2024, 1, 1)
            num_years = end_date.year - start_date.year
            all_dates = []
            views_counter = defaultdict(list)
            headers = {
            'User-Agent': 'MyApp/1.0 (myemail@example.com)',
            'Content-Type': 'application/json',  # Commonly used for POST requests with a body
            }
            for day in range(int(365*num_years)):
                current_date = start_date + timedelta(days=day)
                all_dates.append(current_date.strftime('%Y/%m/%d'))
                
            def get_view(date):
                views = defaultdict(list)
                url = f"https://wikimedia.org/api/rest_v1/metrics/pageviews/top/en.wikipedia/all-access/{date}"
                response = requests.get(url,headers=headers,timeout = 10)
                try:
                    if response.status_code == 200:
                        data = response.json()['items'][0]['articles']
                        if len(data) > 0:
                            for article in data:
                                art = article['article'].replace('_', ' ')
                                views[art].append(article['views'])
                except Exception as e:
                    print (f'error for {date}: {e}')
                time.sleep(0.1)
                return views
            
            all_views_counter = async_process(get_view,all_dates,workers=2,msg = 'Getting views')
            ## Aggregrate the defaultlist
            for vc in all_views_counter:
                for k,v in vc.items():
                    views_counter[k].extend(v)
            
            views_counter = {k:np.mean(v) for k,v in views_counter.items() if len(v) > 0}
            with open(views_path,'wb') as f:
                pickle.dump(views_counter,f)
        else:
            with open(views_path,'rb') as f:
                views_counter = pickle.load(f)
        
        ## After getting views, get related topics according to topic_list
        top_articles = list(views_counter.keys())
        topic_list = list(topic_to_document.keys())
        existing_topics = list(set(top_articles).intersection(set(topic_list)))
        if not os.path.exists(embedding_topic_path):
            embeddings = get_embedding(existing_topics,embedding_topic_path,truncate=False)['all_embeddings']
            with open(embedding_topic_path,'wb') as f:
                pickle.dump(embeddings,f)
        else:
            with open(embedding_topic_path,'rb') as f:
                embeddings = pickle.load(f)
        assert len(embeddings) == len(existing_topics), "Embeddings and topics should be of same length"
        embedding_dict = {'all_embeddings':embeddings,'topics':existing_topics}
        
        num_per_category = (num_topics+num_test_topics)//len(category_list)
        category_topics = get_related_topics(embedding_dict,category_list,num_per_category) # {category:[topics]}
        
        with open(category_topic_path,'wb') as f: # save to fix the topics across different model/runs
            pickle.dump(category_topics,f)
        
    else:
        with open(category_topic_path,'rb') as f:
            category_topics = pickle.load(f)
    
    topic2docu = {}
    for cat,top in category_topics.items():
        for t in top:
            topic2docu[t] = topic_to_document[t]
    ## If need to take test_topics, take out from each category num_test//len(category_list) topics and keep rest for train
    if not os.path.exists(train_test_topics_path):
        test_set = []
        train_topics = deepcopy(category_topics)
        test_topic_per_cat = num_test_topics//len(category_list)
        for cat,cat_topic in category_topics.items():
            acceptable_test_topics = [t for t in cat_topic if len(topic_to_document[t].split()) > (2048/(4/3))] # at least 2048 tokens
            test_region = len(acceptable_test_topics)//5
            if test_region < test_topic_per_cat:
                test_region = test_topic_per_cat*2
            cat_test_topics = random.sample(acceptable_test_topics[:test_region],test_topic_per_cat) # take from the first 20%
            test_set.extend([(cat,t) for t in cat_test_topics]) # take from the first 20% of the topics
            train_topics[cat] = list(set(train_topics[cat]) - set(cat_test_topics))
            
        train_set = []
        for cat,tops in train_topics.items():
            for top in tops:
                train_set.append((cat,top))
        with open(train_test_topics_path,'wb') as f:
            pickle.dump({'train':train_set,'test':test_set},f)
    else:
        with open(train_test_topics_path,'rb') as f:
            train_test_data = pickle.load(f)
        train_set = train_test_data['train']
        test_set = train_test_data['test']
        
    
    return train_set,topic2docu,test_set
    
def get_predefined_topics(num_topics,topic2doc,num_test_topics=0,existing_test_topics=None):
    test_set = []
    train_set = []
    for category,topic_list in predefined_titles.items():
        if existing_test_topics is None:
            selected_test_topics = random.sample(topic_list,num_test_topics)
        else:
            selected_test_topics = existing_test_topics[category]
        test_set.extend([(category,t) for t in selected_test_topics])
        remaining_topics = list(set(topic_list) - set(selected_test_topics))
        train_set.extend([(category,t) for t in remaining_topics])
    
    # train_set = random.sample(train_set,num_topics)
    out_topic2doc = {}
    for _,topic in train_set+test_set:
        document = topic2doc[topic]
        document = clean_document(document) # remove the references
        out_topic2doc[topic] = document
    return train_set,out_topic2doc,test_set


def clean_document(document):
    split_document = document.split('\n')
    end_pos = len(split_document)
    for i,doc in enumerate(split_document):
        if doc.strip().lower() == 'references' or doc.strip().lower() == 'see also':
            end_pos = i
            break
    return '\n'.join(split_document[:end_pos])
    