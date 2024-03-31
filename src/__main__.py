from datetime import datetime
import os
import sys
import spacy
from pyspark.sql import SparkSession
import pyLDAvis.gensim_models

from tf_idf import (
    tf_idf_pipeline_with_saved_model,
    tf_idf_pipeline,
)

from embeddings import (
    embeddings_pipeline,
    embeddings_pipeline_with_saved_response,
)

from topic_modelling import (
    topic_modelling_pipeline,
    get_model_lda_and_k_optimal,
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


def pipeline3(df):
    spacy_nlp = spacy.load("en_core_web_sm")
    lda_model, corpus, dictionary = topic_modelling_pipeline(
        spacy_nlp,
        df,
        no_below=5,
        no_above=0.5,
        keep_n=600,
        workers=13,
        lda_iters=100,
        passes=10,
        num_topics=10,
    )

    utc_datetime = datetime.utcnow()
    utc_datetime_str = utc_datetime.strftime("%Y-%m-%d_%H-%M-%S")

    lda_display = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary)
    pyLDAvis.save_html(lda_display, f"lda_report__{utc_datetime_str}.html")

    # k = get_model_lda_and_k_optimal(
    #     spacy_nlp,
    #     df,
    #     no_below=5,
    #     no_above=0.5,
    #     keep_n=500,
    #     workers=13,
    #     iterations=50,
    #     passes=10,
    #     max_topics=10,
    #     type_coherence="u_mass",
    # )

    # print(k)


def main():
    dataset_path = "/home/sebastian/Downloads/workshop2/climateTwitterData.csv/climateTwitterData.csv"
    spark = SparkSession.builder.master("local[*]").getOrCreate()
    spark.sparkContext.addPyFile(f"{os.getcwd()}/src/tf_idf.py")

    df = spark.read.csv(dataset_path, header=True, inferSchema=True)

    # pipeline1(df)
    # pipeline2(df)
    pipeline3(df)


if __name__ == "__main__":
    sys.exit(main())
