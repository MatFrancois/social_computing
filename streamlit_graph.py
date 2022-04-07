import time
import os
import re
import json
import pandas as pd
import urllib.request
import streamlit as st
import streamlit.components.v1 as components
from gensim.models import Word2Vec
from unidecode import unidecode
# 
# streamlit server side 
# visualiser les communautés:

# Select keywords:

# commu1  | commu2  | commu3  | 
# -----   | -----   | -----   | 
# voisin1 | voisin1 | voisin1 | 
# voisin2 | voisin2 | voisin2 | 
# voisin3 | voisin3 | voisin3 | 
# 

DATA='leaders_per_com_withwv.json'
with open('kept_by_com.json', 'r') as f:
    HASHTAG = json.load(f)
# HASHTAG = dict(filter(lambda x: x[0] in ['22','35','6'], HASHTAG.items()))


st.set_page_config(
 page_title="",
 page_icon="🧊",
 layout="wide",
#  initial_sidebar_state="expanded",
 menu_items={
 'Get Help': 'https://www.extremelycoolapp.com/help',
 'Report a bug': "https://www.extremelycoolapp.com/bug",
 'About': "# This is a header. This is an *extremely* cool app!"
 }
 )


# CSS to inject contained in a string
hide_table_row_index = """
            <style>
            tbody th {display:none}
            .blank {display:none}
            </style>
            """
st.markdown(hide_table_row_index, unsafe_allow_html=True)

@st.cache(allow_output_mutation=True)
def load_models():
    files = os.listdir()
    models = {}
    for file in list(filter(re.compile(r'.*(\.model)$').match, files)):
        i = file.replace('word2vec_com', '').replace('.model', '') # get associated community
        print(f'working on {file}')
        models[i] = file #Word2Vec.load(file)
    print('trigger load model')
    return models

# download models from git 
@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def download_models():
    url = [
        f"https://github.com/GreenAI-Uppa/social_computing/releases/download/models/word2vec_com{i}.model" for i in [22,35,6,2,34,14,13,16,9,5,24,10,31,59,64,0,3,8,11,15,26,29,32,39,40,42,54,70,55,19,46,49,7,39,51,23,25,1,4,66,18,47,12]
    ] + [
        f"https://github.com/GreenAI-Uppa/social_computing/releases/download/models/word2vec_com{i}.model.wv.vectors.npy" for i in [10, 11, 13, 14, 15, 16, 18, 22, 24, 29, 2, 31, 34, 35, 39, 3, 47, 49, 4, 54, 55, 59, 5, 64, 6, 70, 9]
    ] + [
        f"https://github.com/GreenAI-Uppa/social_computing/releases/download/models/word2vec_com{i}.model.syn1neg.npy" for i in [10, 11, 13, 14, 15, 16, 18, 22, 24, 29, 2, 31, 34, 35, 39, 3, 47, 49, 4, 54, 55, 59, 5, 64, 6, 70, 9]
    ] 

    my_bar = st.progress(0)
    delta = 100/len(url)
    for u, i in zip(url, range(len(url))):
        my_bar.progress(int((i+1)*delta))
        filename = u.split('/')[-1]
        if filename in os.listdir(): continue
        try:
            urllib.request.urlretrieve(u, filename)
        except Exception as e:
            print(filename)
            print(u)
            print(e)
        
@st.cache(allow_output_mutation=True)
def load_data(path):
    with open(path, 'r') as f:
        community_details = json.load(f)
    return community_details

# @st.cache(allow_output_mutation=True)
def light_prepro(mot):
    """clean string of accent and useless space

    Args:
        mot (str): string

    Returns:
        str: cleaned string
    """
    return unidecode(mot.lower().strip())

# @st.cache(allow_output_mutation=True)
def get_similar_words(word, model, n, j):
    print('similarity')
    m = Word2Vec.load(model.get(j))
    return  pd.DataFrame(list(map(lambda x: light_prepro(x[0]), m.wv.similar_by_word(word, topn=n))), columns=['Neighbors']) if word in m.wv.key_to_index else None

def get_similar_hashtag(word, model, n, hashtag_dict, j):
    res = {}
    # for k, v in hashtag_dict.items():
    v = hashtag_dict.get(j)
    m = Word2Vec.load(model.get(j))
    print(word)
    if word not in m.wv.key_to_index: return None
    sim = m.wv.distances(word, v)
    print(sim[:20])
    df = pd.DataFrame(v, columns=['Neighbors'])
    df['sim'] = sim
    return df.sort_values('sim', ascending=True).iloc[:n,:]

def leaders_to_df(community_details, cluster_id):
    # print(community_details)
    df = pd.DataFrame.from_dict(community_details.get(cluster_id), orient='index')
    df['username'] = df.index
    return df[['username', 'n_rt']].sort_values(by='n_rt', ascending=False)
 
 
print('starting')

_, col, _ = st.columns(3)
col.title("App Title")
col.markdown(
    '''
Dans quelques jours débutera le premier tour de l'élection présidentielle 2022. Parallèlement, la situation environnementale continue 
de se dégrader et la prise de conscience reste minime. 

## A quoi sert ce site ?

Ce site vous permet d'explorer les différentes communautés politiquement engagées sur twitter et leurs représentants. Vous pourrez alors 
choisir un mot clé et afficher le N termes les plus proches contextuellement à celui ci par communauté. La liste de termes retournée pour 
chaque communauté donne un aperçu des termes utilisés dans un même contexte par les membres de celle ci. 

Des termes renvoyés très différents de celui de référence peuvent signifier une absence de celui ci dans le discours global.

## Méthodologie

Nous avons construit nos communautés au regard des liens formés entre les utilisateurs de twitter et des retweets effectués. L'établissement
des N termes les plus proches contextuellement à un autre a été réalisé grâce à un modèle de langue. Ce modèle se base sur les co-occurences
des mots et permet d'identifier les termes proches. 

Pour chaque communauté un modèle a été entrainé et a permis de d'établir un dictionnaire de terme sous forme de vecteur de taille 300.

## Choisir un mot clé :
    '''
)
# keyword = col.selectbox(label="allowed keyword", options=('nature', 'cop26', 'nucléaire', 'eolien', 'climat', 'musulman')) # prend comme value la première option
keyword = col.text_input(label='',value='climat')

n_voisins = 10 #col.slider('Number of neighbors to display',3, 30, value=10)
n_leaders = 5 #col.slider('Number of leaders to display',2, 50, value=5)
only_hashtag = st.checkbox('Compare to most famous hashtag of community')
# my_bar = st.progress(0)

print(f'n_voisins   :       {n_voisins}')

download_models()

# load communities
community_details = load_data(path=DATA)
# community_details = dict(filter(lambda x: x[0] in ['22','35','6'], community_details.items()))
# load w2v models
models = load_models()
print('model loaded')
buttons = {}

if keyword:

    print(f'keyword     :       {keyword}')
    # sim_dict = get_similar_words(keyword, models, n_voisins) if not only_hashtag else get_similar_hashtag(keyword, models, n_voisins, HASHTAG)
    st.title(f'Mot clé sélectionné : {keyword}')

    compteur = 0
    while compteur < len(community_details):
        
        c = st.container()
        n_col = min(len(community_details)-compteur, 5)

        col = c.columns(n_col)

        for l, co in enumerate(col):
            j = [22,35,6,2,34,14,13,16,9,5,24,10,31,59,64,0,3,8,11,15,26,29,32,39,40,42,54,70,55,19,46,49,7,39,51,23,25,1,4,66,18,47,12][compteur+l] #list(community_details.keys())[compteur-l-1] # à remplacer par l'ordre d'apparition des leaders
            print(j)
            title = f'Community {j}'

            # display leaders
            co.subheader(title)
            with co.expander('leaders', expanded=True):
                st.table(leaders_to_df(community_details, str(j)).iloc[:n_leaders,:])
            print(f'j: {j}')
            df = get_similar_words(keyword, models, n_voisins, str(j)) if not only_hashtag else get_similar_hashtag(keyword, models, n_voisins, HASHTAG, str(j))
            co.table(df)

        st.markdown("""---""")
        compteur += 5

st.markdown('''
## Qui sommes nous ?

L'équipe GreenAI de l'Université de Pau et des Pays de l'Adour est un laboratoire engagé qui améliore les algorithmes d'apprentissage 
automatique de pointe. Soucieux de notre impact sur la planète, nous développons des algorithmes à faible consommation d'énergie et 
relevons les défis environnementaux. Contrairement à d'autres groupes de recherche, nos activités sont dédiées à l'ensemble du pipeline, 
depuis les bases mathématiques jusqu'au prototype de R&D et au déploiement en production avec des partenaires industriels. Nous sommes 
basés à Pau, en France, en face des Pyrénées.            
''')
print('fini')




