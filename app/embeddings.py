import requests
from retry import retry
from json.decoder import JSONDecodeError
import pandas as pd

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"

@retry(JSONDecodeError, tries=3, delay=1)
def mk_query(text, hf_token):
    output = get_embeddings([text], hf_token)
    embeddings_df = pd.DataFrame(output)

    return embeddings_df.values

@retry(JSONDecodeError, tries=3, delay=10)
def get_embeddings(tweets, hf_token):
    ##hf_token = os.environ.get("HF_TOKEN")

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
