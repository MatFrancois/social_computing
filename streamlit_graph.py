import time
import os
import re
import json
import pandas as pd
import urllib.request
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
import locale
from tqdm import tqdm
from sklearn.manifold import TSNE
from gensim.models import Word2Vec
from unidecode import unidecode
import plotly.express as px
import pickle
locale.setlocale(locale.LC_ALL, '')

#---------------------------------- test wordcloud------------------------------------
from PIL import Image
# -----------------------------------------------------------------------------------

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

#with open('communities_length.json', 'r') as f:
#    communities_length = json.load(f)


st.set_page_config(
 page_title="Le climat de la présidentielle",
 page_icon="🧊",
 layout="wide",
#  initial_sidebar_state="expanded",
 menu_items={
 'Report a bug': "https://github.com/GreenAI-Uppa/social_computing/issues",
 'About': '''
L'équipe GreenAIUppa de l'Université de Pau et des Pays de l'Adour est un laboratoire engagé qui améliore 
les algorithmes d'apprentissage automatique de pointe. Soucieux de notre impact sur la planète, nous 
développons des algorithmes à faible consommation d'énergie et relevons les défis environnementaux. 
Contrairement à d'autres groupes de recherche, nos activités sont dédiées à l'ensemble du pipeline, 
depuis les bases mathématiques jusqu'au prototype de R&D et au déploiement en production avec des partenaires 
industriels. Nous sommes basés à Pau, en France, en face des Pyrénées.
 '''
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

# download models from git 
# @st.cache(allow_output_mutation=True, suppress_st_warning=True)
@st.experimental_memo
def download_models_pickle():
    #model_id = [22,35,6] if WORK_ON_ECH else [22,35,6,2,34,14,13,16,9,5,24,10,31,59,64,0,3,8,11,15,26,29,32,39,40,42,54,70,55,19,46,49,7,39,51,23,25,1,4,66,18,47,12]
    # model_id_light = [22,35,6] if WORK_ON_ECH else [10, 11, 13, 14, 15, 16, 18, 22, 24, 29, 2, 31, 34, 35, 39, 3, 47, 49, 4, 54, 55, 59, 5, 64, 6, 70, 9]
    model_id =[87, 32, 18, 66, 45, 81, 32, 19, 15, 88, 74, 73, 50, 22] #[87, 32, 18] #[1, 2, 3, 5, 7, 9, 10, 11, 12, 13, 15, 17, 18, 19, 21, 22, 23, 24, 26, 28, 29, 30, 32, 34, 37, 38, 40, 41, 43, 44, 45, 50, 51, 53, 54, 55, 56, 61, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 79, 80, 81, 83, 84, 85, 86, 87, 88, 89, 104]

    url = [
        f"https://github.com/GreenAI-Uppa/social_computing/releases/download/secondround/word2vec_{i}.pk" for i in model_id
    ] + ['https://github.com/GreenAI-Uppa/social_computing/releases/download/secondround/comm3wow2v.pk']

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
def get_data(com_to_display):
  (com, community_details, hashtags) = pickle.load(open('comm3wow2v.pk','rb'))
  models = {}
  for c in com_to_display:
      models[c]={}
      models[c]['model'] = 'word2vec_'+str(c)+'.pk' # pickle.load(open('word2vec_'+str(c)+'.pk','rb'))
      #models[c]['model'] = pickle.load(open('word2vec_'+str(c)+'.pk','rb'))
  return com, community_details, hashtags, models
 
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
    #m = Word2Vec.load(model.get(j))
    #m = model[int(j)]['model']
    m = pickle.load(open(models[int(j)]['model'],'rb'))
    return  pd.DataFrame(list(map(lambda x: light_prepro(x[0]), m.wv.similar_by_word(word, topn=n))), columns=['Termes']) if word in m.wv.key_to_index else None

def get_similar_hashtag(word, model, n, hashtag_dict, j):
    res = {}
    #v = hashtag_dict.get(j)
    #m = model[int(j)]['model']
    m = pickle.load(open(models[int(j)]['model'],'rb'))
    #m = Word2Vec.load(model.get(j))
    if word not in m.wv.key_to_index: return None
    v = [w for w in hashtag_dict if w in m.wv.key_to_index]
    sim = m.wv.distances(word, v)
    df = pd.DataFrame(list(map(lambda x: f'#{x}', v)), columns=['Termes'])
    df['sim'] = sim
    print('columns',df.columns)
    df.columns = ['Termes voisins','sim']
    return df.sort_values('sim', ascending=True).iloc[:n,:]['Termes voisins']

def leaders_to_df(community_details, cluster_id):
    com = community_details[int(cluster_id)]
    df = pd.DataFrame.from_dict(com, orient='index') #munity_details.get(cluster_id), orient='index')
    #df = pd.DataFrame.from_dict(community_details.get(cluster_id), orient='index')
    df['leaders'] = list(map(lambda x: f'<a target="_blank" href="https://twitter.com/{x}">@{x}</a>', df.index.tolist()))
    df = df[['leaders', 'n_rt']].sort_values(by='n_rt', ascending=False)
    df.columns = ['leaders', 'retweeteurs']
    return df

def get_com_hover_text(models, leaders, n=5):
    texts = []
    for c in models: # to be order consistent with the rest
        df = pd.DataFrame.from_dict(leaders[c], orient='index')
        df['leaders'] = df.index.tolist()
        df = df[['leaders', 'n_rt']].sort_values(by='n_rt', ascending=False)
        df.columns = ['leaders', 'retweeteurs']
        texts.append(' '.join([ '@'+x for x in  df['leaders'].tolist()[:n]]))
    return texts

def jaccard_score(A, B, n=0, verb=False):
    """calcul jaccard_score between A and B dataframe regarding to a selected column and n neighbors

    Args:
        A (pd.DataFrame): first dataframe with neighbors words
        B (pd.DataFrame): second dataframe with neighbors words
        n (int, optional): number of neighbors to calc Jaccard score. Defaults to 0.

    Returns:
        float: return jaccard score
    """
    if n == 0: n = min(len(A), len(B))
    a = set(A)
    b = set(B)
    union = a.union(b) #len(set(a + b))
    inter = a.intersection(b)
    if verb:
        print(inter,len(inter))
    if len(union) == 0:
        return 0
    return len(inter)/len(union)

def concat_similarity_df(df):
    '''
    concat columns of similarity df to consider only on "keywords group"
    '''
    com_key = []
    for i in df:
        com_key += df[i].tolist()

    return list(filter(lambda x: x!='__nokey__', com_key))

def calc_aj(models):
    """calculate average jaccard for multiple keywords and plot matrix with plotly

    Args:
        models (dict): models dict from train_word2vec
        keywords (list): list of keywords (refered to columns in df) to compute aj on
    """
    aj_matrix, distances = [], {}
    for community in tqdm(models):
        ref_keys = concat_similarity_df(df=models.get(community).get('similarity_df'))
        ajs = []
        for to_compare in models:
            comp_keys = concat_similarity_df(df=models.get(to_compare).get('similarity_df'))

            if community == to_compare: ajs.append(1); continue
            sc = (1 - jaccard_score(ref_keys, comp_keys, n=len(ref_keys)))
            ajs.append(sc)
            distances[(community, to_compare)] = sc
            # ajs.append(np.mean([jaccard_score(ref_keys, comp_keys, n=n) for n in range(1,max(len(ref_keys),len(comp_keys)))]))

        aj_matrix.append(ajs)
        # fig = go.Figure(data=go.Heatmap(
        #     z=aj_matrix,
        #     # x=list(models.keys()),
        #     # y=list(models.keys()),
        #     colorscale='Viridis')
        # )
        # fig.show()
    return aj_matrix, distances

def add_similar_words_df(models, keywords, nneigh=30):
    """add similarity df in a model dict and print them

    Args:
        models (dict): models dict from train_word2vec
    """
    for i, value in models.items():
        similarity_df = pd.DataFrame()
        print(f'---{i}---')
        model = pickle.load(open(value['model'],'rb')) # = pickle.load(open('word2vec_'+str(c)+'.pk','rb'))
        for key in keywords:
            try:
                similarity_df[key] = [light_prepro(mot) for mot, _ in model.wv.similar_by_word(key, topn=nneigh)]
            except KeyError:
                similarity_df[key] = ['__nokey__' for _ in range(nneigh)]
        value['similarity_df'] = similarity_df

# =============== Wordcloud =============== #

list_images = os.listdir('wordcloud_hashtag_communities_v3/')
list_images_com = [int(list_images[i][10:12]) for i in range(len(list_images))]
list_com = [87, 32, 18, 66, 45, 81, 88, 74, 73, 50, 22]

# ========================================= #        
        
print('starting')

_, col, _ = st.columns([1,3,1])
col.title("Le Climat de la Présidentielle")
col.markdown(
    '''
L'éléction présidentielle bat son plein! Parallèlement, la situation environnementale continue de se 
dégrader et la prise de conscience reste minime.

## A quoi sert ce site ?

Ce site vous permet d'explorer les différentes communautés politiquement et/ou écologiquement engagées 
sur twitter et de comparer leurs champs lexicaux par rapport à des sujets de votre choix. Concrètement, 
il vous est proposé de choisir plusieurs mots clés afin d'observer ces champs sont voisins d'une communauté à l'autre. D'une part en représentant les communautés sur un graphique, et d'autre part, en affichant N termes contextuellement voisins pour chaque 
communauté. Autrement dit, ces listes de termes donnent un aperçu du lexique utilisé dans le contexte du 
mot clé pour chaque communauté (voir une [démo vidéo](https://youtu.be/IedJytgIFE0) de 4 minutes).

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

Nous contacter : [Matthieu François](mailto:matthieu.francois@yahoo.fr)
''', unsafe_allow_html=True)

# selecting a subset of the communities for clarity and to save ram
com_to_display = [22,35,6,2,34,14,13,16,9,5,24,10,31,59,64,0,3,8,11,15,26,29,32,39,40,42,54,70,55,19,46,49,7,39,51,23,25,1,4,66,18,47,12]
com_to_display = [81, 32, 18, 66, 45, 87, 88, 74, 73, 50, 22]
### loading all the data

download_models_pickle()
com, community_details, hashtags, models = get_data(com_to_display)

if False:
    (com, community_details, hashtags) = pickle.load(open('data/comm3wow2v.pk','rb'))
    models = {}
    for c in com_to_display:
        models[c]={}
        models[c]['model'] = pickle.load(open('data/word2vec_'+str(c)+'.pk','rb'))

#(com, text_users, df, community_details, models, leaders, hashtags) = pickle.load(open('/home/paul/data/elyzee/comm_and_co3_subset_leadersFormat.pk','rb')) #comm_and_co3_subset.pk','rb'))
communities_length = dict([(str(c), len(com[c])) for c in models])

col.markdown('''### Choisir un mot clé (ou voir [démo](https://youtu.be/IedJytgIFE0))''') 

# keyword = col.selectbox(label="allowed keyword", options=('nature', 'cop26', 'nucléaire', 'eolien', 'climat', 'musulman')) # prend comme value la première option
keyword = col.text_input(label='',value='GIEC')


n_voisins = 10 #col.slider('Number of neighbors to display',3, 30, value=10)
n_leaders = 5 #col.slider('Number of leaders to display',2, 50, value=5)

print(f'n_voisins   :       {n_voisins}')

#download_models()
# load communities
#community_details = load_data(path=DATA)
if WORK_ON_ECH: community_details = dict(filter(lambda x: x[0] in ['22','35','6'], community_details.items()))
# load w2v models
#models = load_models()
print('model loaded')
buttons = {}
# display communities & words
if keyword:
    print(f'keyword     :       {keyword}')
    st.subheader(f'Communautés (leaders et nuage de mots) + termes voisins du mot clé : {keyword}')

    only_hashtag = st.checkbox('Cocher pour restreindre les termes à des Hashtags', value=True)
    compteur = 0
    while compteur < len(com_to_display): #len(community_details):

        n_col = min(len(com_to_display)-compteur, 5)
        col = st.columns(n_col)

        for l, co in enumerate(col):
            if compteur+l >= len(com_to_display):
                continue
            j = com_to_display[compteur+l] #list(community_details.keys())[compteur-l-1] # à remplacer par l'ordre d'apparition des leaders
            title = f'Communauté {j} (taille : {communities_length.get(str(j)):n})'
            # display leaders
            # co.subheader(title)
            df = get_similar_hashtag(keyword, models, n_voisins, hashtags, str(j)) if only_hashtag else get_similar_words(keyword, models, n_voisins, str(j))
            with co.expander(title, expanded=True):
                st.markdown(
                    f'''{leaders_to_df(community_details, str(j)).iloc[:n_leaders,:].to_html(escape=False, index=False)}''', unsafe_allow_html=True)
                st.write('')    
                if j in list_images_com:
                    paths = 'wordcloud_hashtag_communities_v3'+f'/community_{j}.png'
                    im = Image.open(paths)
                    st.image(im, caption=f'Wordcloud de la communauté {j}') 
            print(f'j: {j}')
            co.table(df)
        st.markdown("""---""")
        compteur += 5

    list_leaders = ['enedis','EnviroMag','INRAE_France','M_Laigneau','CompteurLinky','emma_ducros','TristanKamin','fmomboisse','MacLesggy','GeWoessner','J_Bardella','RNational_off', 
        'MLP_officiel','SansLui_','BrunoBilde','ZemmourEric','stanislasrig','F_Desouche','GilbertCollard','christine_kelly','EmmanuelMacron','JeanCASTEX','Elysee','avecvous','RichardFerrand',
        'audreygarric','BonPote','yjadot','franceinter','Reporterre','JLMelenchon','MathildePanot','melenchon_2022','ManonAubryFr','Ugobernalicis','idrissaberkane','PhilippeMurer','DIVIZIO1',
        'france_soir','CStrateges','lemondefr','ArianeChemin','benvtk','libe','RaphaelleBacque','le_gorafi','bertrandhadet','AvocatavecunE','Ganette_',
        'Decimaitre','ONU_fr','ONUinfo','UNESCO_fr','CCNUCC','ONUGeneve']
    lead = st.selectbox("Sélectionnez l'un des 55 leaders présenté ci-dessous pour découvrir son nuage de mots associés", list_leaders)
    # lead_images = os.listdir('/home/jpalafox/social_computing/social_computing/wordcloud_images_per_users')
    lead_images = 'wordcloud_images_per_users'
    lead_image = Image.open(lead_images + f'/{lead}.png')
    st.image(lead_image, caption=f"Wordcloud de l'utilisateur {lead}")
print('fini')




