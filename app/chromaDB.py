import chromadb
from datetime import datetime
from chromadb.utils import embedding_functions

from embeddings import (
    mk_query,
)

def get_collection(chroma_db_path, collection_name):
    client = chromadb.PersistentClient(path=chroma_db_path)
    collection = client.get_or_create_collection(
    name=collection_name, embedding_function=embedding_functions.DefaultEmbeddingFunction())

    return collection, client

def query(collection, search_text, n_results, hf_token):
    query = mk_query(search_text, hf_token)
    result = collection.query(
        query_embeddings=query,
        n_results=n_results,
    )

    return result

def get_n_tweets(collection, search_text, n_results, hf_token):
    collection_count = collection.count()
    if collection_count < n_results:
        n_results = collection_count

    return query(collection, search_text, n_results, hf_token)

def get_all_tweets(collection, search_text, hf_token):    
    return query(collection, search_text, collection.count(), hf_token)

def print_collection(result, n_results):
    print("###########################\n")
    for idx in range(0, n_results):
        print(f"ID: {result['ids'][0][idx]}, Document: {result['documents'][0][idx]}, Distance: {result['distances'][0][idx]}")
        print("###########################\n")

def save_result(result, n_results):
    utc_datetime = datetime.utcnow()
    utc_datetime_str = utc_datetime.strftime("%Y-%m-%d_%H-%M-%S")
    file = open(f"./results__{utc_datetime_str}.txt", "w")
    
    for idx in range(0, n_results):
        file.write(f"ID: {result['ids'][0][idx]}, Document: {result['documents'][0][idx]}, Distance: {result['distances'][0][idx]}\n")

    file.close()