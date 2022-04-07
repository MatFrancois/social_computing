import contextlib
import json, os, re
from tqdm import tqdm
from pprint import pprint
import spacy
'''
- lecture de tous les tweets
- PAS DE RELATION JAIME PRISE EN COMPTE
'''


ROOT='/data/datasets/elyzee_2022'
WRITE_RELATION = True

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


def extract_relation(files, relation_file=None, write_relation=False, nlp=None):
    relation = []
    users = {}
    for file in tqdm(files):
        with open(file, 'r') as f:
            try:
                tweet = json.load(f)
            except Exception:
                continue
            create_relation(relation, tweet, write_relation, relation_file)
            extract_user_and_text(users, tweet, nlp)
    return relation, users 

def create_relation(relation, tweet, save, relation_file):
    author = tweet.get('author').get('username')
    identifiant = tweet.get('id')
    if re.search('^RT @(.*?):\s', tweet.get('text')):
        relation.append([identifiant, re.findall(r'^RT @(.*?):\s', tweet.get('text'))[0], author, 'rt'])
        if save: relation_file.write(f"{identifiant};{re.findall(r'^RT @(.*?): ', tweet.get('text'))[0]};{author};rt\n") # write data
    else:
        with contextlib.suppress(AttributeError, TypeError):
            for mention in tweet.get('entities').get('mentions'):
                muser = mention.get('username')
                if muser is not None:
                    relation.append([identifiant, author, muser, 'mention'])
                    if save:
                        relation_file.write(f"{identifiant};{author};{muser};mention\n") # write data
            
def extract_user_and_text(users, tweet, nlp):
    if not re.search('^RT @(.*?):\s', tweet.get('text')):
        author = tweet.get('author').get('username')

        text_tk = [doc.lemma_ for doc in nlp(tweet.get('text')) if not doc.is_stop and not doc.is_punct and not re.match(r'https?://', doc.text)]
        if author in users:
            users[author]['text'].append(text_tk)
            users[author]['text_root'].append(tweet.get('text'))
        else:
            users[author] = {
                'text': [text_tk],
                'text_root': [tweet.get('text')]
            }

def main():
    files = list_files(path=ROOT) 
    nlp = spacy.load('fr_core_news_sm')
    
    if WRITE_RELATION: relation_file = open('/data/mfrancois/social_computing/relations.txt', 'w')

    relation, users = extract_relation(files, relation_file=relation_file, write_relation=True, nlp=nlp) # get and write relation
    
    with open('/data/mfrancois/social_computing/community_details.json', 'w') as f:
        json.dump(users, f)
        
        
    print(f'longueur relation: {len(relation)}')
    
    if WRITE_RELATION: relation_file.close()
    
    
if __name__ == "__main__":
    main()





