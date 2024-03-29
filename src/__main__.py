import os
import sys
import spacy
from pyspark.sql import SparkSession

from tf_idf import (
    tokenize_tweet,
    tokenize_dataset,
    compute_tf_idf_model,
    model_to_tf_idf,
    load_tf_idf_model,
    tf_idf_pipeline_with_saved_model,
    tf_idf_pipeline,
)


def pipeline1():
    dataset_path = "/home/sebastian/Downloads/workshop2/climateTwitterData.csv/climateTwitterData.csv"
    spacy_nlp = spacy.load("en_core_web_sm")
    spark = SparkSession.builder.master("local[*]").getOrCreate()
    spark.sparkContext.addPyFile(f"{os.getcwd()}/src/tf_idf.py")

    df = spark.read.csv(dataset_path, header=True, inferSchema=True)

    tf_idf_pipeline(spacy_nlp, df)
    # tf_idf_pipeline_with_saved_model("./4d56eb56-84f4-46fe-a505-44d0c4cd3ffa", spacy_nlp, df)


if __name__ == "__main__":
    pipeline1()
    sys.exit(0)
