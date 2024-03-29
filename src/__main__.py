import os
import sys
import spacy
from pyspark.sql import SparkSession

from tf_idf import (
    tf_idf_pipeline_with_saved_model,
    tf_idf_pipeline,
)

from embeddings import (
    embeddings_pipeline,
    embeddings_pipeline_with_saved_response,
)


def pipeline1(df):
    spacy_nlp = spacy.load("en_core_web_sm")
    # tf_idf_pipeline(spacy_nlp, df)
    tf_idf_pipeline_with_saved_model(
        "./4d56eb56-84f4-46fe-a505-44d0c4cd3ffa", spacy_nlp, df
    )


def pipeline2(df):
    embeddings_pipeline(df, 3000, "./chromadb")
    # embeddings_pipeline_with_saved_response(
    #     df, 3000, "embeddings__2024-03-29_22-02-54.json", "./chromadb"
    # )


def main():
    dataset_path = "/home/sebastian/Downloads/workshop2/climateTwitterData.csv/climateTwitterData.csv"
    spark = SparkSession.builder.master("local[*]").getOrCreate()
    spark.sparkContext.addPyFile(f"{os.getcwd()}/src/tf_idf.py")

    df = spark.read.csv(dataset_path, header=True, inferSchema=True)

    # pipeline1(df)
    pipeline2(df)


if __name__ == "__main__":
    sys.exit(main())
