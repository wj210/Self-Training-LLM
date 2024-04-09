from sentence_transformers import SentenceTransformer
from scipy.spatial import KDTree
import torch
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
    all_embeddings = torch.tensor(all_embeddings)
    all_embeddings = torch.nn.functional.normalize(all_embeddings, p=1, dim=-1)
    return KDTree(all_embeddings)

def get_embedding(topic_list,embedding_path):
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2", device="cuda")
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
    import faiss
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2", device="cpu")
    all_embeddings = embedding_dict['all_embeddings']
    topic_list = embedding_dict['topics']
    
    assert num_topics >0, "num_topics should be greater than 0"
    topic_embedding_space = faiss.IndexFlatIP(all_embeddings.shape[-1])
    faiss.normalize_L2(all_embeddings)
    topic_embedding_space.add(all_embeddings)
    all_selected_topics = []
    embedded_query_topics = embedder.encode(query_topics, device="cpu", convert_to_numpy=True)
    for i,query_vec in tqdm(enumerate(embedded_query_topics),total = len(embedded_query_topics),desc = 'finding related topics'):
        non_duplicate_topics = []
        k = deepcopy(num_topics)
        num_tries = 0
        query_vec = np.array([query_vec],dtype=np.float32)
        faiss.normalize_L2(query_vec)
        while len(non_duplicate_topics) < num_topics:
            # emb_distance,emb_indices = topic_embedding_space.query(query_embedding,k=k,workers=16)
            _, emb_indices = topic_embedding_space.search(query_vec, k)
            selected_topics = [topic_list[i] for i in emb_indices[0]]
            for t in selected_topics:
                if t not in set(all_selected_topics + non_duplicate_topics):
                    non_duplicate_topics.append(t)
                if len(non_duplicate_topics) >= num_topics:
                    break
            num_tries += 1
            k *= 2
            if num_tries > 5:
                print (f'Only found {len(non_duplicate_topics)} topics for {query_topics[i]}. Required: {num_topics}')
                break
        all_selected_topics.extend(non_duplicate_topics)
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

def get_top_articles(num_topics,topic_to_document,num_test_topics=0,embedding_dict=None):
    views_path = 'data/embeddings/wikiviews.pkl'
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
    sorted_views = sorted(views_counter.items(), key=lambda item: item[1], reverse=True)
    out_topics = []
    topic2docu = {}
    for topic,_ in tqdm(sorted_views,total=len(sorted_views),desc = 'Getting top articles'):
        if topic in topic_to_document:
            out_topics.append(topic)
            topic2docu[topic] = topic_to_document[topic]
        if len(out_topics) >= num_topics:
            break
        
    if embedding_dict is not None and num_test_topics > 0:
        embeddings = embedding_dict['all_embeddings']
        topic_list = embedding_dict['topics']
        existing_topics = set([t for t,_ in sorted_views]).intersection(set(topic_list))
        top_topics = [t for t in existing_topics if t not in set(out_topics)] # get the top topics that are not in out_topics
        selected_idx = [topic_list.index(t) for t in top_topics]
        topic_list = [topic_list[i] for i in selected_idx]
        embeddings = embeddings[selected_idx]
        embedding_dict = {'all_embeddings':embeddings,'topics':topic_list}
        query_topics = out_topics[:num_test_topics]
        test_topics = get_related_topics(embedding_dict,query_topics,num_topics=1)
        for t in test_topics:
            topic2docu[t] = topic_to_document[t]
    else:
        test_topics = []
    
    return out_topics,topic2docu,test_topics
    
    