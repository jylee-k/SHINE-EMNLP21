from clean_str import clean_str
import tqdm
import nltk
import json

def NER(inputs = None, ent2id_new_path = None, stop_word = None, split='train'):
    
    ent2id_new = json.load(open(ent2id_new_path, 'r'))
    adj_ent_index = []
    query_nodes = []
    tag_set = set()
    entity_set = set()
    words_set = set()
    idx = []
    coarse_labels = []
    fine_labels = []
    #labels = []
    tag_list = []
    word_list = []
    ent_mapping = {} 
    for i, item in enumerate(tqdm(inputs)):
        # item=f_train[str(i)]
        query = clean_str(item['text'])
        if not query:
            print(query)
            continue
        tags = [one[1].lower() for one in nltk.pos_tag(nltk.word_tokenize(query))]
        if '' in tags:
            print(item)

        tag_list.append(' '.join(tags))
        tag_set.update(tags)
        #labels.append(item['label'])
        coarse_labels.append(item['coarse_label'])
        fine_labels.append(item['fine_label'])
        if stop_word:
            words = [one.lower() for one in query.split(' ') if one not in stop_word]
        else:
            words = [one.lower() for one in query.split(' ')]
        if '' in words:
            print(words)

        ent_list = []
        index = [] 
        for key in ent2id_new.keys():
            if key in query.lower():
                ent_list.append(key)
                if key not in ent_mapping:
                    ent_mapping[key] = len(ent_mapping)
                    entity_set.update(ent_list)
                if ent_mapping[key] not in index: index.append(ent_mapping[key])
        # print(entity_set)
        adj_ent_index.append(index)
        word_list.append(' '.join(words))
        words_set.update(words)
        if query:
            query_nodes.append(query)
        else:
            print(item)
            print(query)
        idx.append(len(idx))