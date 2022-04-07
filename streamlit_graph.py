import time
import os
import re
import json
import pandas as pd
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

# PRESENTATION




DATA='/data/mfrancois/social_computing/leaders_community.json'
MODELS='/data/mfrancois/social_computing/models'
TEST='/data/mfrancois/social_computing/toto.json'
@st.cache(allow_output_mutation=True)
def load_models(path):
    files = os.listdir(path)
    models = {}
    for file in list(filter(re.compile(r'.*(\.model)$').match, files)):
        i = file.replace('word2vec_com', '').replace('.model', '') # get associated community
        models[i] = Word2Vec.load(f'{path}/{file}')
    print('trigger load model')
    return models

@st.cache(allow_output_mutation=True)
def load_data(path):
    with open(path, 'r') as f:
        community_details = json.load(f)
    return community_details

@st.cache(allow_output_mutation=True)
def light_prepro(mot):
    """clean string of accent and useless space

    Args:
        mot (str): string

    Returns:
        str: cleaned string
    """
    return unidecode(mot.lower().strip())

@st.cache(allow_output_mutation=True)
def get_similar_words(word, model, n):
    return {m: pd.DataFrame(list(map(lambda x: light_prepro(x[0]), v.wv.similar_by_word(word, topn=n))), columns=['Neighbors']) for m, v in model.items() if word in v.wv.key_to_index}
    
print('starting')

keyword = st.text_input(label='Choose keyword',value='climat')
n_voisins = st.number_input('Choose Number of neighbors', min_value=3, max_value=30, value=10)

print(f'n_voisins   :       {n_voisins}')

# load communities
community_details = load_data(path=DATA)
# load w2v models
models = load_models(path=MODELS)
buttons = {}

if keyword:

    print(f'keyword     :       {keyword}')
    sim_dict = get_similar_words(keyword, models, n_voisins)

    st.title(keyword)

    compteur = 1
    while compteur != len(models):
        
        c = st.container()
        n_col = min(len(models)-compteur, 3)

        col = c.columns(n_col)

        for l, co in enumerate(col):
            j = list(models.keys())[compteur-l-1]
            title = f'community {j}'
            with co.expander(title, expanded=True):
                # buttons[j] = st.button('know more', key=str(j))
                st.table(sim_dict.get(j))

        compteur += 3            

# for key, val in buttons.items():
#     if val:
#         print(f'community n°{key} details asked')
#         st.sidebar.text(community_details.get(key))
side = st.sidebar.container()
side.select_slider()
# leaders = {}
# for k, v in buttons.items():
#     if v:
#         leaders[v] = get_community(community_details, cluster_id=k, nb_user=5)
#         st.sidebar.text(leaders.get(v))

print('fini')




