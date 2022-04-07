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

DATA='leaders_community.json'
MODELS=''

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
def load_models(path):
    files = os.listdir(path)
    models = {}
    for file in list(filter(re.compile(r'.*(\.model)$').match, files)):
        i = file.replace('word2vec_com', '').replace('.model', '') # get associated community
        models[i] = Word2Vec.load(f'{path}/{file}')
    print('trigger load model')
    return models

# download models from git 
@st.cache(allow_output_mutation=True)
def download_models():
    url = [
        'https://github.com/GreenAI-Uppa/social_computing/releases/download/models/leaders_community.json',
        "https://github.com/GreenAI-Uppa/social_computing/releases/download/models/word2vec_com27.model",
        "https://github.com/GreenAI-Uppa/social_computing/releases/download/models/word2vec_com27.model.syn1neg.npy",
        "https://github.com/GreenAI-Uppa/social_computing/releases/download/models/word2vec_com27.model.wv.vectors.npy"
    ]
    my_bar = st.progress(0)
    delta = 100/len(url)
    for u, i in zip(url, range(len(url))):
        my_bar.progress((i+1)*delta)
        filename = u.split('/')[-1]
        urllib.request.urlretrieve(u, filename)





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
def get_similar_words(word, model, n):
    print('similarity')
    return {m: pd.DataFrame(list(map(lambda x: light_prepro(x[0]), v.wv.similar_by_word(word, topn=n))), columns=['Neighbors']) for m, v in model.items() if word in v.wv.key_to_index}

def leaders_to_df(community_details, cluster_id):
    # print(community_details)
    df = pd.DataFrame.from_dict(community_details.get(cluster_id), orient='index')
    df['username'] = df.index
    return df[['username', 'edges']].sort_values(by='edges', ascending=False)
 
 
print('starting')

_, col, _ = st.columns(3)
col.title("App Title")
col.markdown(
    ''' > *Présentation des la méthode de formation de communauté et de l'entrainement de n word2vec pour n communauté*
    '''
)
col.image('images/louvain_algo.png')
keyword = col.selectbox(label="allowed keyword", options=('nature', 'cop26', 'nucléaire', 'eolien', 'climat', 'musulman')) # prend comme value la première option
# keyword = col.text_input(label='Choose keyword',value='climat')

n_voisins = col.slider('Number of neighbors to display',0, 30, value=10)
# my_bar = st.progress(0)

print(f'n_voisins   :       {n_voisins}')

download_models()

# load communities
community_details = load_data(path=DATA)
# load w2v models
models = load_models(path=MODELS)
print('model loaded')
print(models.get('27').wv.similar_by_word('climat'))
buttons = {}

if keyword:

    print(f'keyword     :       {keyword}')
    sim_dict = get_similar_words(keyword, models, n_voisins)

    st.title(keyword)

    compteur = 0
    while compteur < len(community_details):
        
        c = st.container()
        n_col = min(len(community_details)-compteur, 5)

        col = c.columns(n_col)

        for l, co in enumerate(col):
            j = list(community_details.keys())[compteur-l-1] # à remplacer par l'ordre d'apparition des leaders
            title = f'Community {j}'
            
            # display leaders
            co.subheader(title)
            with co.expander('leaders'):
                st.table(leaders_to_df(community_details, j))
            # co.markdown("""---""")
            co.table(sim_dict.get(j))
        
        st.markdown("""---""")
        compteur += 5


print('fini')




