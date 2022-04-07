import json
'''
Ecriture des 10 leaders de chaque communauté contenues dans community_details_with_id.json pour gagner du temps sur le dashboard streamlit
'''


PATH='/data/mfrancois/social_computing/community_details_with_id.json'
PATH_TO_WRITE='/data/mfrancois/social_computing/leaders_community.json'

def get_data(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def write_data(path, data):
    with open(path, 'w') as f:
        json.dump(data, f)
        

def get_community(community_details, cluster_id, nb_user=5):
    # filter on id
    print('start get community')
    filtered_community = dict(filter(lambda x: x[1].get('cluster') == cluster_id and len(x[1].get('text'))>0, community_details.items()))
    # sort on leader
    edges  = [v.get('edges') for v in filtered_community.values()]
    edges.sort(reverse=True)
    try:
        filtered_community = dict(filter(lambda x: x[1].get('edges') >= edges[4], filtered_community.items()))
    except IndexError:
        print(f'removing cluster n°{cluster_id}, because too few publications')
        return None
    return dict(sorted(filtered_community.items(), key=lambda x: x[1].get('edges')))


def main():
    data = get_data(PATH)
    leaders = {i: [] for i in {val.get('cluster') for val in data.values()}}
    leaders_data = {i: get_community(data, i, 10) for i in leaders}

    print(f'nb leaders: {len(leaders_data)}')
    leaders_data = dict(filter(
        lambda x: x[1] is not None,
        leaders_data.items()
    ))
    print(f'nb leaders: {len(leaders_data)}')

    write_data(PATH_TO_WRITE, leaders_data)

if __name__ == "__main__":
    main()
