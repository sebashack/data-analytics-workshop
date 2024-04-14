import chromadb
from chromadb.utils import embedding_functions

from embeddings import (
    mk_query,
)

def create_client(chroma_db_path):
    return chromadb.PersistentClient(path=chroma_db_path)

def get_or_create_collection(client, collection_name):
    return client.get_or_create_collection(
                name=collection_name, 
                embedding_function=embedding_functions.DefaultEmbeddingFunction()
    )

def get_tweet_by_id(ids, collection):
    if ids == None:
        ids = range(0, count_collection(collection))
    return collection.get(
        ids=ids,
    )

def count_collection(collection):
    return collection.count()

def query(collection, search_text, hf_token, n_results):
    query_embeddings = mk_query(search_text, hf_token)
    return collection.query(
        query_embeddings=query_embeddings,
        n_results=n_results,
        where_document={"$contains": f"{search_text}"}
    )