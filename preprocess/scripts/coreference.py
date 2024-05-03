import coreferee, spacy, spacy_transformers

# !python -m spacy download en_core_web_trf
# !python -m spacy download en_core_web_lg
# python -m coreferee install en


def coreference(sentence):
    nlp = spacy.load('en_core_web_trf')
    nlp.add_pipe('coreferee')
    doc = nlp(sentence)
    # TODO: parse the coreference annotations into a graph (sparse or dense?)
    # can be accessed via doc._.coref_chains

