from sentence_transformers import SentenceTransformer
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')

def word_similarity():
    docs = ['hi', 'hello', 'how are you']
    docs_embeddings = embedding_model.encode(docs)
    word_embeddings = embedding_model.encode(docs, output_value="token_embeddings")

    token_ids = []
    token_strings = []
    tokenizer = embedding_model._first_module().tokenizer

    for doc in docs: 
        ids = tokenizer.encode(doc)
        strings = tokenizer.convert_ids_to_tokens(ids)
        token_ids.append(ids)
        token_strings.append(strings)