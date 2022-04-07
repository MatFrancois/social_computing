import json
import os
import networkx as nx
import networkx.algorithms.community as nx_comm
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import spacy
import unidecode
import pandas as pd
import numpy as np
from tqdm import tqdm
from gensim.models import Word2Vec
from gensim.models.phrases import Phraser, Phrases
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords

stop_words =set(stopwords.words('french'))
tk = TweetTokenizer()
nlp = spacy.load('fr_core_news_sm')
keywords = ['climat', 'énergie', 'nature']
PATH='/data/mfrancois/social_computing/relations.txt'
ROOT='/data/datasets/elyzee_2022/10:01:2021-00:00:00_03:22:2022-00:00:00'


def list_files(path):
    """get all files in path & subdirectories

    Args:
        path (str): path

    Returns:
        list: path files list
    """
    files = []
    for element in os.listdir(path):
        if os.path.isdir(f'{path}/{element}'):
            files.extend(iter(list_files(f'{path}/{element}')))
            continue
        files.append(f'{path}/{element}')
    return files


def read_data(path, select='all'):
    """read relation save before

    Args:
        path (str): data path
        select (str, optional): filter data on relation type ('rt' or 'mention' or 'all'). Defaults to 'all'.

    Returns:
        list: edges relation like [
            [user1, user2],
            [user1, user3],
            ...
        ]
    """
    with open(path, 'r') as f:
        data = [line.replace('\n', '').split(';')[1:] for line in f.readlines()]
    if select in ['rt', 'mention']:
        return list(filter(lambda x: x[2] == select, data))
    return data

def create_graph(data, save=False, path='graph.gml'):
    """create networkx graph

    Args:
        data (list): relation data list from read_data
        save (bool, optional): save graph or not. Defaults to False.
        path (str, optional): path to save if save on true. Defaults to 'graph.gml'.

    Returns:
        nx.Graph: Graph
    """
    g = nx.Graph()
    for edge in data: g.add_edge(*edge)
    if save: nx.write_gml(g, path)
    return g

def create_community(g, return_json=True):
    """create community based on Louvain algorithm

    Args:
        g (nx.Graph): graph data
        return_json (bool, optional): return community in a json. Defaults to True.

    Returns:
        dict: comminuty details as {
            user: {
                edges: number of edges,
                cluster: community id,
            }
        }
    """
    com = nx_comm.louvain_communities(g)
    print(f'nb com: {len(com)}')

    print('identifying leaders..')
    print(f'{"name":<20}|{"edges":<20}| commu size')
    print('-'*50)
    
    if return_json:
        community_details = {}
        for i, cluster in enumerate(com):
            if len(cluster)<500: continue
            user_inside_community = []
            for user in cluster:
                community_details[user] = {
                    'edges': len(g[user]),
                    'cluster': i,
                    'text': [],
                    'text_root': []
                }
                user_inside_community.append([user, len(g[user])])
            name, edges = max(user_inside_community, key=lambda x: x[1])

            print(f'{name:<20}|{edges:<20,}| {len(cluster):,}')
        return com, community_details
    return com
    

def add_text_to_community(community_details, text_users, save=False):
    '''add text for each user in community_details
    
    Args:
        community_details (dict): dictionnary of user from create_community()
        text_users (dict): dictionnary from files with {user: {text: lemmatized text, text_root: original text}}
    '''
    for name in community_details:
        if name in text_users:
            community_details[name]['text'] = text_users.get(name).get('text')
            community_details[name]['text_root'] = text_users.get(name).get('text_root')
    if save: 
        with open('/data/mfrancois/social_computing/community_details_with_id.json', 'w') as f:
            json.dump(community_details, f)
    
def json_to_df(community_details):
    """convert community_details to dataframe

    Args:
        community_details (dict): community_dict from create_community

    Returns:
        pd.DataFrame: dataframe with username, number of edges and community id
    """
    users = list(community_details.keys())
    edges = [community_details.get(user).get('edges') for user in users]
    clusters = [community_details.get(user).get('cluster') for user in users]

    df = pd.DataFrame()
    df['username'] = users
    df['n_edges'] = edges
    df['i'] = clusters
    return df

def train_word2vec(df, community_details, save=False):
    """train word2vec skipgram

    Args:
        df (pd.DataFrame): dataframe created with json_to_df()
        community_details (dict): community_details from create_community
        save (bool, optional): save model or not. Defaults to False.

    Returns:
        dict: dictionnary with each model by community {community_number: model}
    """
    models = {}
    for i in tqdm(df.i.unique()):
        
        try:
            text_by_community = [txt for user in df.username[df['i'] == i] for txt in community_details.get(user).get('text') if len(txt)>0]
        except AttributeError as e:
            print(e)
            code = input()
            while code != 'stop':
                exec(code)
                code = input()

        bigram = Phrases(text_by_community, min_count=35, threshold=2,delimiter=' ')

        bigram_phraser = Phraser(bigram)
        bigram_token = [bigram_phraser[sen] for sen in text_by_community]
        model = Word2Vec(bigram_token, min_count=1,vector_size= 300,workers=5, window =5, sg = 1)
        models[i] = {'model': model}
        
        if save: model.save(f"/data/mfrancois/social_computing/models/word2vec_com{i}.model")
        
    return models

def light_prepro(mot):
    """clean string of accent and useless space

    Args:
        mot (str): string

    Returns:
        str: cleaned string
    """
    return unidecode.unidecode(mot.lower().strip())

def add_similar_words_df(models):
    """add similarity df in a model dict and print them

    Args:
        models (dict): models dict from train_word2vec
    """
    for i, value in models.items():
        similarity_df = pd.DataFrame()
        print(f'---{i}---')
        for key in keywords:
            try:
                similarity_df[key] = [light_prepro(mot) for mot, _ in value.get('model').wv.similar_by_word(key, topn=30)]
            except KeyError:
                similarity_df[key] = ['__nokey__' for _ in range(30)]
        print(f'    {keywords[0]:<40}|    {keywords[1]:<40}|    {keywords[2]:<40}')
        print('-'*134)
        for j in range(similarity_df.shape[0]): print(f'    {similarity_df[keywords[0]][j]:<40}|    {similarity_df[keywords[1]][j]:<40}|    {similarity_df[keywords[2]][j]:<40}')
        value['similarity_df'] = similarity_df
        
def jaccard_score(A, B, column, n=0):
    """calcul jaccard_score between A and B dataframe regarding to a selected column and n neighbors

    Args:
        A (pd.DataFrame): first dataframe with neighbors words
        B (pd.DataFrame): second dataframe with neighbors words
        column (str): column to compare
        n (int, optional): number of neighbors to calc Jaccard score. Defaults to 0.

    Returns:
        float: return jaccard score
    """
    if A[column][0] == '__nokey__' or B[column][0] == '__nokey__': return 0 # le mot n'est jamais cité par la communauté
    if n == 0: n = min(len(A), len(B))
    a = A[column].tolist()[:n]
    b = B[column].tolist()[:n]
    union = len(set(a + b))
    inter = len([e for e in a if e in b])
    return inter/union

def calc_aj_and_plot(models, df, keywords, save=True):
    """calculate average jaccard for multiple keywords and plot matrix with plotly

    Args:
        models (dict): models dict from train_word2vec
        df (pd.DataFrame): df from json_to_df
        keywords (list): list of keywords (refered to columns in df) to compute aj on
        save (bool, optional): save plotly image. Defaults to True.
    """
    for key in keywords:
        aj_matrix = []
        for community in models:
            ajs = []
            for to_compare in models:
                if community == to_compare: ajs.append(1); continue
                ajs.append(np.mean([jaccard_score(models.get(community).get('similarity_df'), models.get(to_compare).get('similarity_df'), column=key, n=n) for n in range(1,31)]))
            aj_matrix.append(ajs)

        print('ploting..')
        leaders = [df[df.i==comu_i][df[df.i==comu_i].n_edges == max(df[df.i==comu_i].n_edges)].username.tolist()[0] for comu_i in df.i.unique()]

        fig = go.Figure(data=go.Heatmap(
                        z=aj_matrix,
                        x=leaders,
                        y=leaders,
                        colorscale='Viridis'))
        fig.update_layout(
            title=f'AJ score for {key}',
            xaxis_nticks=36)

        if save: fig.write_image(f"/data/mfrancois/social_computing/aj4{key}.png")



def main():
    data = read_data(path=PATH, select='all')
    
    # keep only relation
    print('========= Manage graph =========')
    data = list(map(lambda x: x[:2], data))
    g = create_graph(data, save=True, path='/data/mfrancois/social_computing/graph.gml')
    _, community_details = create_community(g)
    print(len(community_details))
    
    # get user and text (lighter)
    with open('/data/mfrancois/social_computing/community_details.json', 'r') as f:
        text_users = json.load(f)
        
    print('====== Manage communities ======')
    add_text_to_community(community_details=community_details, text_users=text_users, save=True)
        
    # train word embedding
    df = json_to_df(community_details=community_details)
    print('====== Training WordToVec ======')
    models = train_word2vec(df=df, community_details=community_details, save=True)
    
    print('===== Similarity and Plots =====')
    add_similar_words_df(models=models)    

    # average jaccard score
    calc_aj_and_plot(models=models, df=df, keywords=keywords, save=True)        


if __name__ == "__main__":
    main()


