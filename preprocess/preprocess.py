import json
import numpy as np
import pickle as pkl
import math
import nltk
from tqdm import tqdm
import os
import spacy
import textacy
from vp_patterns import VP_PATTERNS

from scipy.sparse import coo_matrix
import matplotlib.pyplot as plt
import re
from sentence_transformers import SentenceTransformer

# embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2")
# nltk.download("averaged_perceptron_tagger")
nlp = spacy.load('en_core_web_lg')


def clean_str(string, use=True):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    if not use:
        return string

    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " 's", string)
    string = re.sub(r"\'ve", " 've", string)
    string = re.sub(r"n\'t", " n't", string)
    string = re.sub(r"\'re", " 're", string)
    string = re.sub(r"\'d", " 'd", string)
    string = re.sub(r"\'ll", " 'll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_stopwords(filepath="./stopwords_en.txt"):
    stopwords = set()
    with open(filepath, "r") as f:
        for line in f:
            swd = line.strip()
            stopwords.add(swd)
    print(len(stopwords))
    return stopwords


def tf_idf_transform(inputs, mapping=None, sparse=False):
    """
    +    Apply TF-IDF transformation to the input data.
    +
    +    Args:
    +        inputs (list): Input data.
    +        mapping (dict or None, optional): Mapping of terms to feature indices. Default is None.
    +        sparse (bool, optional): Whether to return sparse matrix. Default is False.
    +
    +    Returns:
    +        array-like or scipy.sparse.coo_matrix: Transformed input data with TF-IDF weights.
    """

    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.feature_extraction.text import TfidfTransformer
    from scipy.sparse import coo_matrix

    vectorizer = CountVectorizer(vocabulary=mapping)
    tf_idf_transformer = TfidfTransformer()
    tf_idf = tf_idf_transformer.fit_transform(vectorizer.fit_transform(inputs))
    weight = tf_idf.toarray()
    return weight if not sparse else coo_matrix(weight)


def PMI(inputs, mapping, window_size, sparse):
    """
    inputs (list): list of strings

    """
    W_ij = np.zeros([len(mapping), len(mapping)], dtype=np.float64)
    W_i = np.zeros([len(mapping)], dtype=np.float64)
    W_count = 0
    for one in inputs:
        word_list = one.split(" ")
        if len(word_list) - window_size < 0:
            window_num = 1
        else:
            window_num = len(word_list) - window_size + 1
        for i in range(window_num):
            W_count += 1
            context = list(set(word_list[i : i + window_size]))
            while "" in context:
                context.remove("")
            for j in range(len(context)):
                W_i[mapping[context[j]]] += 1
                for k in range(j + 1, len(context)):
                    W_ij[mapping[context[j]], mapping[context[k]]] += 1
                    W_ij[mapping[context[k]], mapping[context[j]]] += 1
    if sparse:
        rows = []
        columns = []
        data = []
        for i in range(len(mapping)):
            rows.append(i)
            columns.append(i)
            data.append(1)
            tmp = [ele for ele in np.nonzero(W_ij[i])[0] if ele > i]
            for j in tmp:
                value = math.log(W_ij[i, j] * W_count / W_i[i] / W_i[j])
                if value > 0:
                    rows.append(i)
                    columns.append(j)
                    data.append(value)
                    rows.append(j)
                    columns.append(i)
                    data.append(value)
        PMI_adj = coo_matrix(
            (data, (rows, columns)), shape=(len(mapping), len(mapping))
        )
    else:
        PMI_adj = np.zeros([len(mapping), len(mapping)], dtype=np.float64)
        for i in range(len(mapping)):
            PMI_adj[i, i] = 1
            tmp = [ele for ele in np.nonzero(W_ij[i])[0] if ele > i]
            # for j in range(i + 1, len(mapping)):
            for j in tmp:
                pmi = math.log(W_ij[i, j] * W_count / W_i[i] / W_i[j])
                if pmi > 0:
                    PMI_adj[i, j] = pmi
                    PMI_adj[j, i] = pmi
    return PMI_adj

def get_phrases(text):

    doc = nlp(text)

    verb_phrases = textacy.extract.token_matches(doc, VP_PATTERNS)


    phrase_list = []

    tag_list = []


    # Extract Noun Phrases and corresponding pos tags

    for chunk in doc.noun_chunks:

        phrase_list.append(chunk.text)

        tag_list.append("".join([t.pos_ for t in chunk]))

    # Print all Verb Phrase and corresponding pos tags

    for chunk in verb_phrases:

        phrase_list.append(chunk.text)

        tag_list.append("".join([t.pos_ for t in chunk]))

    return phrase_list, tag_list


def make_node2id_eng_text(dataset_name, remove_StopWord=False):

    stop_word = load_stopwords()
    stop_word.add("")
    os.makedirs(f"./{dataset_name}_data", exist_ok=True)

    # load text and labels
    f_train = json.load(open("./{}_split.json".format(dataset_name)))["train"]
    f_test = json.load(open("./{}_split.json".format(dataset_name)))["test"]

    from collections import defaultdict

    word_freq = defaultdict(int)
    for item in f_train.values():  # item is a text-label pair
        # clean the text and split by whitespace to get words
        words = clean_str(item["text"]).split(" ")
        # count the word frequency
        for one in words:
            word_freq[one.lower()] += 1
    for item in f_test.values():  # item is a text-label pair
        # clean the text and split by whitespace to get words
        words = clean_str(item["text"]).split(" ")
        # count the word frequency
        for one in words:
            word_freq[one.lower()] += 1

    # filter words with frequency < 5
    freq_stop = 0
    for word, count in word_freq.items():
        if count < 5:
            stop_word.add(word)
            freq_stop += 1
    print("freq_stop num", freq_stop)

    ent2id_new = json.load(open("../pretrained_emb/NELL_KG/ent2ids_refined", "r"))
    adj_ent_index = []
    query_nodes = []
    tag_set = set()
    phrase_tag_set = set()
    entity_set = set()
    words_set = set()
    phrases_set = set()
    joined_phrases_set = set()
    train_idx = []
    test_idx = []
    coarse_labels = []
    fine_labels = []
    labels = []
    tag_list = []
    phrase_tag_list = []
    word_list = []
    phrase_list = []
    joined_phrase_list = []
    ent_mapping = {}

    # iterate through train set
    for i, item in enumerate(tqdm(f_train.values())):
        # clean text
        query = clean_str(item["text"])
        # check if text is empty after cleaning
        if not query:
            print(query)
            continue
        # get pos tags for words in the query
        tags = [one[1].lower() for one in nltk.pos_tag(nltk.word_tokenize(query))]
        if "" in tags:
            print(item)
            
        # get pos tags for phrases in the query
        phrases, phrase_tags = get_phrases(query)
        if "" in phrase_tags:
            print(item)

        # join the tags to form a tag string
        tag_list.append(" ".join(tags))
        # update the tag set with unseen tags
        tag_set.update(tags)
        
        # join the phrase tags to form a tag string
        phrase_tag_list.append(" ".join(phrase_tags))
        # update the phrase tag set with unseen tags
        phrase_tag_set.update(phrase_tags)
        
        # append labels
        # labels.append(item["label"])

        coarse_labels.append(item['coarse_label'])
        fine_labels.append(item['fine_label'])

        # get list of words from text
        if remove_StopWord:
            words = [one.lower() for one in query.split(" ") if one not in stop_word]
        else:
            words = [one.lower() for one in query.split(" ")]
        if "" in words:
            print(words)

        # named entity recognition
        ent_list = []
        index = []
        # for every word in the NER dictionary
        for key in ent2id_new.keys():
            # check if the word is in the text
            if key in query.lower():
                # add word to entity list
                ent_list.append(key)
                # check if word is already in the mapping dict
                if key not in ent_mapping:
                    # add word to mapping dict in order
                    ent_mapping[key] = len(ent_mapping)
                    # update the entity set with new words from sentence
                    entity_set.update(ent_list)

                if ent_mapping[key] not in index:
                    # index: a list of entities present in the sentence
                    index.append(ent_mapping[key])
        # print(entity_set)
        # entity adjacency matrix: list[list] of entities present in the sentences
        adj_ent_index.append(index)
        # word_list: list of sentences
        word_list.append(" ".join(words))
        # update word set
        words_set.update(words)
        
        phrase_list.append(" ".join(phrases))
        # update the phrase set with unseen phrases
        phrases_set.update(phrases)
        
        # joined phrases
        joined_phrases_set.update(["_".join(phrase.split()) for phrase in phrases])
        joined_phrase_list.append(" ".join(["_".join(phrase.split()) for phrase in phrases]))
        
        assert len(phrases_set) == len(joined_phrases_set), print(phrases, ["_".join(phrase.split()) for phrase in phrases])
        
        if query:
            query_nodes.append(query)
        else:
            print(item)
            print(query)
        train_idx.append(len(train_idx))

    # iterate through the test set
    for i, item in enumerate(tqdm(f_test.values())):
        # item = f_test[str(i)]
        query = clean_str(item["text"])
        # print(query)
        if not query:
            print(query)
            continue
        tags = [one[1].lower() for one in nltk.pos_tag(nltk.word_tokenize(query))]
        
         # get pos tags for phrases in the query
        phrases, phrase_tags = get_phrases(query)
        if "" in phrase_tags:
            print(item)
        
        tag_list.append(" ".join(tags))
        tag_set.update(tags)
        
        # join the phrase tags to form a tag string
        phrase_tag_list.append(" ".join(phrase_tags))
        # update the phrase tag set with unseen tags
        phrase_tag_set.update(phrase_tags)
        
        # labels.append(item["label"])
        coarse_labels.append(item['coarse_label'])
        fine_labels.append(item['fine_label'])
        if remove_StopWord:
            words = [one.lower() for one in query.split(" ") if one not in stop_word]
        else:
            words = [one.lower() for one in query.split(" ")]
        if "" in words:
            print(words)

        # named entity recognition
        ent_list = []
        index = []
        # for every word in the NER dictionary
        for key in ent2id_new.keys():
            # check if the word is in the text
            if key in query.lower():
                # check if word is already in the mapping dict
                if key not in ent_mapping:
                    # add word to ent_list
                    ent_list.append(key)
                    # add word to mapping dict as word:idx_in_ent_list
                    ent_mapping[key] = len(ent_mapping)
                    # update the set
                    entity_set.update(ent_list)
                if ent_mapping[key] not in index:
                    index.append(ent_mapping[key])
        # entity adjacency (index) matrix: list[list] of entities present in the sentences
        adj_ent_index.append(index)

        word_list.append(" ".join(words))
        words_set.update(words)
        
        # update the phrase set with unseen phrases
        phrase_list.append(" ".join(phrases))
        phrases_set.update(phrases)
        
        # joined phrases
        joined_phrases_set.update(["_".join(phrase.split()) for phrase in phrases])
        joined_phrase_list.append(" ".join(["_".join(phrase.split()) for phrase in phrases]))
        
        assert len(phrases_set) == len(joined_phrases_set), print(phrases, ["_".join(phrase.split()) for phrase in phrases])
        if query:
            query_nodes.append(query)
        else:
            print(item)
            print(query)

        test_idx.append(len(test_idx) + len(train_idx))
    assert len(phrase_list) == len(joined_phrase_list)
    assert len(phrases_set) == len(joined_phrases_set)

    print(tag_set)
    json.dump(
        [adj_ent_index, ent_mapping],
        open("./{}_data/index_and_mapping.json".format(dataset_name), "w"),
        ensure_ascii=False,
    )
    ent_emb = []
    TransE_emb_file = np.loadtxt("../pretrained_emb/NELL_KG/entity2vec.TransE")
    TransE_emb = []

    for i in range(len(TransE_emb_file)):
        TransE_emb.append(list(TransE_emb_file[i, :]))

    rows = []
    data = []
    columns = []

    max_num = len(ent_mapping)
    # creating a coo format for matrix of adj_ent_index
    for sent_i, indices in enumerate(adj_ent_index):
        for index in indices:
            data.append(1)
            rows.append(sent_i)
            columns.append(index)

    # create a matrice of ones and zeros
    # ones correspond to (sentence_index, entity_index) i.e. which entities are present in the sentence
    adj_ent = coo_matrix((data, (rows, columns)), shape=(len(adj_ent_index), max_num))
    # for entity in entity mapping
    for key in ent_mapping.keys():
        # add embedding to ent_emb
        ent_emb.append(TransE_emb[ent2id_new[key]])

    ent_emb = np.array(ent_emb)
    print("ent shape", ent_emb.shape)
    ent_emb_normed = ent_emb / np.sqrt(np.square(ent_emb).sum(-1, keepdims=True))
    adj_emb = np.matmul(ent_emb_normed, ent_emb_normed.transpose())
    print("entity_emb_cos", np.mean(np.mean(adj_emb, -1)))
    pkl.dump(
        np.array(ent_emb), open("./{}_data/entity_emb.pkl".format(dataset_name), "wb")
    )
    pkl.dump(adj_ent, open("./{}_data/adj_query2entity.pkl".format(dataset_name), "wb"))

    word_nodes = list(words_set)
    tag_nodes = list(tag_set)
    phrase_nodes = list(phrases_set)
    phrase_tag_nodes = list(phrase_tag_set)
    entity_nodes = list(entity_set)
    joined_phrases_nodes = list(joined_phrases_set)
    # nodes_all = list(query_nodes | tag_nodes | entity_nodes)
    nodes_all = query_nodes + tag_nodes + entity_nodes + word_nodes + phrase_nodes + phrase_tag_nodes
    nodes_num = len(query_nodes) + len(tag_nodes) + len(entity_nodes) + len(word_nodes) + len(phrase_nodes) + len(phrase_tag_nodes)
    print("query", len(query_nodes))
    print("tag", len(tag_nodes))
    print("ent", len(entity_nodes))
    print("word", len(word_nodes))
    print("phrase", len(phrase_nodes))
    print("phrase_tag", len(phrase_tag_nodes))

    if len(nodes_all) != nodes_num:
        print("duplicate name error")

    print("len_train", len(train_idx))
    print("len_test", len(test_idx))
    print("len_queries", len(query_nodes))

    tags_mapping = {key: value for value, key in enumerate(tag_nodes)}
    words_mapping = {key: value for value, key in enumerate(word_nodes)}
    adj_query2tag = tf_idf_transform(tag_list, tags_mapping)
    adj_tag = PMI(tag_list, tags_mapping, window_size=5, sparse=False)
    
    phrase_tags_mapping = {key: value for value, key in enumerate(phrase_tag_nodes)}
    phrases_mapping = {key: value for value, key in enumerate(phrase_nodes)}
    joined_phrases_mapping = {key: value for value, key in enumerate(joined_phrases_nodes)}
    adj_query2phrase_tag = tf_idf_transform(phrase_tag_list, phrase_tags_mapping)
    adj_phrase_tag = PMI(phrase_tag_list, phrase_tags_mapping, window_size=5, sparse=False)
    
    pkl.dump(
        adj_query2tag, open("./{}_data/adj_query2tag.pkl".format(dataset_name), "wb")
    )
    pkl.dump(adj_tag, open("./{}_data/adj_tag.pkl".format(dataset_name), "wb"))
    pkl.dump(adj_query2phrase_tag, open("./{}_data/adj_query2phrase_tag.pkl".format(dataset_name), "wb"))
    pkl.dump(adj_phrase_tag, open("./{}_data/adj_phrase_tag.pkl".format(dataset_name), "wb"))
    
    adj_query2word = tf_idf_transform(word_list, words_mapping, sparse=True)
    adj_word = PMI(word_list, words_mapping, window_size=5, sparse=True)
    adj_query2phrase = tf_idf_transform(phrase_list, phrases_mapping, sparse=True)
    adj_phrase = PMI(joined_phrase_list, joined_phrases_mapping, window_size=5, sparse=True)

    pkl.dump(
        adj_query2word, open("./{}_data/adj_query2word.pkl".format(dataset_name), "wb")
    )
    pkl.dump(adj_word, open("./{}_data/adj_word.pkl".format(dataset_name), "wb"))
    pkl.dump(adj_query2phrase, open("./{}_data/adj_query2phrase.pkl".format(dataset_name), "wb"))
    pkl.dump(adj_phrase, open("./{}_data/adj_phrase.pkl".format(dataset_name), "wb")) #

    json.dump(
        train_idx,
        open("./{}_data/train_idx.json".format(dataset_name), "w"),
        ensure_ascii=False,
    )
    json.dump(
        test_idx,
        open("./{}_data/test_idx.json".format(dataset_name), "w"),
        ensure_ascii=False,
    )

    # label_map = {value: i for i, value in enumerate(set(labels))}
    # json.dump(
    #     [label_map[label] for label in labels],
    #     open("./{}_data/labels.json".format(dataset_name), "w"),
    #     ensure_ascii=False,
    # )
    coarse_label_map = {value: i for i, value in enumerate(set(coarse_labels))}
    json.dump([coarse_label_map[coarse_label] for coarse_label in coarse_labels], open('./{}_data/coarse_labels.json'.format(dataset_name), 'w'),
              ensure_ascii=False)
    fine_label_map = {value: i for i, value in enumerate(set(fine_labels))}
    json.dump([fine_label_map[fine_label] for fine_label in fine_labels], open('./{}_data/fine_labels.json'.format(dataset_name), 'w'),
              ensure_ascii=False)
    json.dump(
        query_nodes,
        open("./{}_data/query_id2_list.json".format(dataset_name), "w"),
        ensure_ascii=False,
    )
    json.dump(
        tag_nodes,
        open("./{}_data/tag_id2_list.json".format(dataset_name), "w"),
        ensure_ascii=False,
    )
    json.dump(
        entity_nodes,
        open("./{}_data/entity_id2_list.json".format(dataset_name), "w"),
        ensure_ascii=False,
    )
    json.dump(
        word_nodes,
        open("./{}_data/word_id2_list.json".format(dataset_name), "w"),
        ensure_ascii=False,
    )
    json.dump(
        phrase_nodes,
        open("./{}_data/phrase_id2_list.json".format(dataset_name), "w"),
        ensure_ascii=False,
    )
    json.dump(
        phrase_tag_nodes,
        open("./{}_data/phrase_tag_id2_list.json".format(dataset_name), "w"),
        ensure_ascii=False,
    )

    # word embeddings
    glove_emb = pkl.load(open("../pretrained_emb/old_glove_6B/embedding_glove.p", "rb"))
    vocab = pkl.load(open("../pretrained_emb/old_glove_6B/vocab.pkl", "rb"))
    word_embs = []
    err_count = 0
    for word in word_nodes:
        if word in vocab:
            word_embs.append(glove_emb[vocab[word]])
        else:
            err_count += 1
            # print('error:', word)
            word_embs.append(np.zeros(300, dtype=np.float64))
    print("unknown word in glove embedding", err_count)
    pkl.dump(
        np.array(word_embs, dtype=np.float64),
        open("./{}_data/word_emb.pkl".format(dataset_name), "wb"),
    )
    
    # phrase embeddings
    phrasebert_model = SentenceTransformer("Deehan1866/finetuned-phrase-bert-large")
    phrase_embs = []
    for phrase in phrase_nodes:
        phrase_embs.append(phrasebert_model.encode(phrase))
    assert len(phrase_embs) == len(phrase_nodes)
    assert len(phrase_embs) == adj_phrase.shape[0], print(len(phrase_embs), adj_phrase.shape[0])
    pkl.dump(
        np.array(phrase_embs, dtype=np.float64),
        open("./{}_data/phrase_emb.pkl".format(dataset_name), "wb"),
    )
    
    


dataset_name = "trec"
if dataset_name in ["mr", "snippets", "tagmynews"]:
    remove_StopWord = True
else:
    remove_StopWord = False
make_node2id_eng_text(dataset_name, remove_StopWord)
