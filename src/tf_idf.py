import uuid
from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, StringType
from pyspark.ml.feature import CountVectorizer, IDF
from pyspark.ml import Pipeline, PipelineModel


def tf_idf_pipeline(spacy_nlp, df):
    tokenized_df = tokenize_dataset(df, spacy_nlp)

    tokenized_df.select("tokenized_text").show(10, truncate=False)

    model = compute_tf_idf_model(tokenized_df, vocab_size=20)

    model.save(f"./{str(uuid.uuid4())}")

    vocabulary, tfidf_df = model_to_tf_idf(model, tokenized_df)

    print(vocabulary)
    tfidf_df.select("tfidf_features").show(10, truncate=False)


def tf_idf_pipeline_with_saved_model(model_dir_path, spacy_nlp, df):
    tokenized_df = tokenize_dataset(df, spacy_nlp)

    model = load_tf_idf_model(model_dir_path)

    vocabulary, tfidf_df = model_to_tf_idf(model, tokenized_df)

    print(vocabulary)
    tfidf_df.select("tfidf_features").show(10, truncate=False)


def tokenize_tweet(tweet, spacy_nlp):
    doc = spacy_nlp(tweet.lower())
    return [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]


def tokenize_dataset(df, spacy_nlp):
    tweets_df = df.select("text")

    tokenize_tweet_udf = udf(
        lambda s: tokenize_tweet(s, spacy_nlp), ArrayType(StringType())
    )
    tweets_df = df.select("text")
    tokenized_df = tweets_df.withColumn("tokenized_text", tokenize_tweet_udf("text"))

    return tokenized_df


def compute_tf_idf_model(tokenized_df, vocab_size):
    cv = CountVectorizer(
        inputCol="tokenized_text",
        outputCol="tf_features",
        vocabSize=vocab_size,
        minDF=1.0,
    )
    idf = IDF(inputCol="tf_features", outputCol="tfidf_features")

    pipeline = Pipeline(stages=[cv, idf])

    model = pipeline.fit(tokenized_df)

    return model


def model_to_tf_idf(model, tokenized_df):
    tfidf_df = model.transform(tokenized_df)

    vocabulary = model.stages[0].vocabulary

    return vocabulary, tfidf_df


def load_tf_idf_model(path):
    model = PipelineModel.load(path)

    return model
