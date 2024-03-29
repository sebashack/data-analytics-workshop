import os
import uuid
import sys
import spacy
from pyspark.sql import SparkSession

from tf_idf import (
    tokenize_tweet,
    tokenize_dataset,
    compute_tf_idf_model,
    model_to_tf_idf,
    load_tf_idf_model,
)


def tf_idf_pipeline():
    dataset_path = "/home/sebastian/Downloads/workshop2/climateTwitterData.csv/climateTwitterData.csv"
    spacy_nlp = spacy.load("en_core_web_sm")
    spark = SparkSession.builder.master("local[*]").getOrCreate()
    spark.sparkContext.addPyFile(f"{os.getcwd()}/src/tf_idf.py")

    df = spark.read.csv(dataset_path, header=True, inferSchema=True)

    tokenized_df = tokenize_dataset(df, spacy_nlp)

    # tokenized_df.select("tokenized_text").show(10, truncate=False)

    # model = compute_tf_idf_model(tokenized_df, vocab_size=20)

    # model.save(f"./{str(uuid.uuid4())}")

    model = load_tf_idf_model("./4d56eb56-84f4-46fe-a505-44d0c4cd3ffa")

    vocabulary, tfidf_df = model_to_tf_idf(model, tokenized_df)

    print(vocabulary)
    tfidf_df.select("tfidf_features").show(10, truncate=False)


if __name__ == "__main__":
    tf_idf_pipeline()
    sys.exit(0)
