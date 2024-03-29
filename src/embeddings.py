import numpy as np
import chromadb
from datetime import datetime
import json
import os
import requests
from retry import retry
from json.decoder import JSONDecodeError
import pandas as pd


MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"


def embeddings_pipeline(df, n_tweets, chroma_db_path):
    rows = df.select("text").collect()

    tweets = [row.text for row in rows]

    output = get_embeddings(tweets[:n_tweets])

    utc_datetime = datetime.utcnow()
    utc_datetime_str = utc_datetime.strftime("%Y-%m-%d_%H-%M-%S")

    if output is not None:
        with open(f"./embeddings__{utc_datetime_str}.json", "w") as file:
            file.write(json.dumps(output))

    embeddings_df = pd.DataFrame(output)

    client = chromadb.PersistentClient(path=chroma_db_path)

    collection = create_collection(client, tweets[:n_tweets], embeddings_df)

    query = mk_query("climate people")

    result = collection.query(
        query_embeddings=query,
        n_results=10,
    )

    print(result)


def embeddings_pipeline_with_saved_response(df, n_tweets, json_path, chroma_db_path):
    rows = df.select("text").collect()

    tweets = [row.text for row in rows]

    data = None

    with open(json_path, "r") as file:
        data = json.load(file)

    embeddings_df = pd.DataFrame(data)

    client = chromadb.PersistentClient(path=chroma_db_path)

    collection = create_collection(client, tweets[:n_tweets], embeddings_df)

    query = mk_query("climate people")

    result = collection.query(
        query_embeddings=query,
        n_results=10,
    )

    print(result)


def create_collection(client, tweets, embeddings_df):
    collection = client.create_collection(name="tweet_collection")

    collection.add(
        embeddings=embeddings_df.values,
        documents=tweets,
        metadatas=[{"type": "tweet"} for _ in range(0, len(tweets))],
        ids=[str(i) for i in range(0, len(tweets))],
    )

    return collection


@retry(JSONDecodeError, tries=3, delay=1)
def mk_query(text):
    output = get_embeddings([text])
    embeddings_df = pd.DataFrame(output)

    return embeddings_df.values


@retry(JSONDecodeError, tries=3, delay=10)
def get_embeddings(tweets):
    hf_token = os.environ.get("HF_TOKEN")

    if hf_token is None:
        raise ValueError("Environment variable HF_TOKEN is not defined")

    api_url = (
        f"https://api-inference.huggingface.co/pipeline/feature-extraction/{MODEL_ID}"
    )
    headers = {"Authorization": f"Bearer {hf_token}"}

    response = requests.post(
        api_url,
        headers=headers,
        json={"inputs": tweets, "options": {"wait_for_model": True}},
    )

    print("Hugging Face API Status Code:", response.status_code)

    if response.status_code != 200:
        raise RuntimeError("Hugging Face request failed")

    return response.json()
