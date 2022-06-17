# Le Climat de la Présidentielle

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

## Qui sommes Nous ? 

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

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/matfrancois/social_computing/main/streamlit_graph.py)

## Projet

- `community_details.json` : Json contenant pour chaque individu le ou les textes extraits, token, etc...
- `graphe.py` 
  - Création des communautés selon le fichier `relations.txt` 
  - Préprocessing & écriture des outputs
  - Création des modèles de w2v par communauté
  - Calcul des distances jaccard
- `load_json.py` : Lecture de N fichiers (contenant chacun les informations d'un tweet) et création de `community_details.json` et `relations.txt`
- `print_first.py` : Lecture de tweets 1 par 1 dans l'ordre d'apparition
- `relations.txt` : Fichier csv à 2 colonnes contenant les relations entre individus selon les retweets / mentions
- `streamlit_graph.py`: Script d'application streamlit dédiée
