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

WORK_ON_ECH = False # if true : work on only 3 models

DATA='leaders_per_com_withwv.json'
with open('kept_by_com.json', 'r') as f:
    HASHTAG = json.load(f)
if WORK_ON_ECH: HASHTAG = dict(filter(lambda x: x[0] in ['22','35','6'], HASHTAG.items()))

with open('communities_length.json', 'r') as f:
    communities_length = json.load(f)


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
    model_id = [22,35,6] if WORK_ON_ECH else [22,35,6,2,34,14,13,16,9,5,24,10,31,59,64,0,3,8,11,15,26,29,32,39,40,42,54,70,55,19,46,49,7,39,51,23,25,1,4,66,18,47,12]
    model_id_light = [22,35,6] if WORK_ON_ECH else [10, 11, 13, 14, 15, 16, 18, 22, 24, 29, 2, 31, 34, 35, 39, 3, 47, 49, 4, 54, 55, 59, 5, 64, 6, 70, 9]
    url = [
        f"https://github.com/GreenAI-Uppa/social_computing/releases/download/models/word2vec_com{i}.model" for i in model_id
    ] + [
        f"https://github.com/GreenAI-Uppa/social_computing/releases/download/models/word2vec_com{i}.model.wv.vectors.npy" for i in model_id_light
    ] + [
        f"https://github.com/GreenAI-Uppa/social_computing/releases/download/models/word2vec_com{i}.model.syn1neg.npy" for i in model_id_light
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
    return  pd.DataFrame(list(map(lambda x: light_prepro(x[0]), m.wv.similar_by_word(word, topn=n))), columns=['Termes']) if word in m.wv.key_to_index else None

def get_similar_hashtag(word, model, n, hashtag_dict, j):
    res = {}
    # for k, v in hashtag_dict.items():
    v = hashtag_dict.get(j)
    m = Word2Vec.load(model.get(j))
    if word not in m.wv.key_to_index: return None
    sim = m.wv.distances(word, v)
    df = pd.DataFrame(list(map(lambda x: f'#{x}', v)), columns=['Termes'])
    df['sim'] = sim
    return df.sort_values('sim', ascending=True).iloc[:n,:]['Termes']

def leaders_to_df(community_details, cluster_id):
    # print(community_details)
    df = pd.DataFrame.from_dict(community_details.get(cluster_id), orient='index')
    # df['username'] = list(map(lambda x: f'@{x}', df.index.tolist()))
    df['leaders'] = list(map(lambda x: f'<a target="_blank" href="https://twitter.com/{x}">@{x}</a>', df.index.tolist()))
    
    df = df[['leaders', 'n_rt']].sort_values(by='n_rt', ascending=False)
    df.columns = ['leaders', 'retweets totaux']
    return df
 
 
print('starting')

_, col, _ = st.columns([1,3,1])
col.title("App Title")
col.markdown(
    '''
L'éléction présidentielle bat son plein! Parallèlement, la situation environnementale continue de se 
dégrader et la prise de conscience reste minime.

## A quoi sert ce site ?

Ce site vous permet d'explorer les différentes communautés politiquement et/ou écologiquement engagées 
sur twitter et de comparer leurs champs lexicaux par rapport à des sujets de votre choix. Concrètement, 
il vous est proposé de choisir un mot clé afin d'afficher N termes contextuellement voisins pour chaque 
communauté. Autrement dit, ces listes de termes donnent un aperçu du lexique utilisé dans le contexte du 
mot clé pour chaque communauté.

Note: vous pourrez être surpris par des voisins très différents de votre mot clé. Cela correspond souvent 
à une absence de celui-ci dans les discussions de cette communauté.


## Méthodologie
**Les données** : Environ 8 millions de tweets ont été collectés entre octobre 2021 et mars 2022. Ils 
correspondent à 227 256 comptes issus d'une liste d'une centaine de politiciens et d'écologistes; à 
ceux-ci s'ajoutent l'extraction automatique de leur followers, les comptes qui les retweetent et mentionnent.

**Algorithme** : Une détection automatique des communautés a été effectuée en considérant qu'un retweet 
établit un lien de proximité entre deux comptes. Chaque communauté est décrit par ses "leaders", c'est 
à dire ses membres ayant accumulé le plus de retweets. Les distances entre le mot clé et les voisins se 
basent sur des statistiques de co-occurences entre les mots : deux mots accompagnés souvent des mêmes 
termes seront considérés voisins.
''')
with col.expander('En savoir plus sur notre équipe'):
    st.markdown('''
L'équipe GreenAIUppa de l'Université de Pau et des Pays de l'Adour est un laboratoire engagé qui améliore 
les algorithmes d'apprentissage automatique de pointe. Soucieux de notre impact sur la planète, nous 
développons des algorithmes à faible consommation d'énergie et relevons les défis environnementaux. 
Contrairement à d'autres groupes de recherche, nos activités sont dédiées à l'ensemble du pipeline, 
depuis les bases mathématiques jusqu'au prototype de R&D et au déploiement en production avec des partenaires 
industriels. Nous sommes basés à Pau, en France, en face des Pyrénées.         

<center>
    <img src="https://miro.medium.com/max/700/0*X36NgC4u0VJBQwF6.png"  alt="centered image" style="text-align: center;">
</center>

[Visiter notre page](https://greenai-uppa.github.io/) 
''', unsafe_allow_html=True)

col.markdown('''## Choisir un mot clé :''')

# keyword = col.selectbox(label="allowed keyword", options=('nature', 'cop26', 'nucléaire', 'eolien', 'climat', 'musulman')) # prend comme value la première option
keyword = col.text_input(label='',value='climat')

n_voisins = 10 #col.slider('Number of neighbors to display',3, 30, value=10)
n_leaders = 5 #col.slider('Number of leaders to display',2, 50, value=5)
only_hashtag = st.checkbox('Cocher pour restreindre les termes à des Hashtags')
# my_bar = st.progress(0)

print(f'n_voisins   :       {n_voisins}')

download_models()

# load communities
community_details = load_data(path=DATA)
if WORK_ON_ECH: community_details = dict(filter(lambda x: x[0] in ['22','35','6'], community_details.items()))
# load w2v models
models = load_models()
print('model loaded')
buttons = {}

# display communities & words
if keyword:

    print(f'keyword     :       {keyword}')
    # sim_dict = get_similar_words(keyword, models, n_voisins) if not only_hashtag else get_similar_hashtag(keyword, models, n_voisins, HASHTAG)
    st.title(f'Mot clé sélectionné : {keyword}')

    compteur = 0
    while compteur < len(community_details):
        
        n_col = min(len(community_details)-compteur, 5)

        col = st.columns(n_col)

        for l, co in enumerate(col):
            j = [22,35,6,2,34,14,13,16,9,5,24,10,31,59,64,0,3,8,11,15,26,29,32,39,40,42,54,70,55,19,46,49,7,39,51,23,25,1,4,66,18,47,12][compteur+l] #list(community_details.keys())[compteur-l-1] # à remplacer par l'ordre d'apparition des leaders
            title = f'Communité n°{j} (taille : {communities_length.get(str(j))})'

            # display leaders
            # co.subheader(title)
            with co.expander(title, expanded=True):
                st.markdown(
                    f'''{leaders_to_df(community_details, str(j)).iloc[:n_leaders,:].to_html(escape=False, index=False)}''', unsafe_allow_html=True)
                # st.table(leaders_to_df(community_details, str(j)).iloc[:n_leaders,:])
                # display_leaders(leaders_to_df(community_details, str(j)).iloc[:n_leaders,:])
            print(f'j: {j}')
            df = get_similar_hashtag(keyword, models, n_voisins, HASHTAG, str(j)) if only_hashtag else get_similar_words(keyword, models, n_voisins, str(j))
            co.table(df)
        st.markdown("""---""")
        compteur += 5


print('fini')




