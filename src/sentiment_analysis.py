from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from pyspark.sql.functions import when, monotonically_increasing_id
from sklearn.feature_extraction.text import CountVectorizer
from tf_idf import (
    tokenize_tweet,
)


def sentiment_analysis_pipeline(df, spacy_nlp):
    df_with_id = df.withColumn("row_id", monotonically_increasing_id())
    unclassified_df = df_with_id.filter(df_with_id.row_id > 30000)

    classified_df = df.limit(
        30000
    )  # Only get the first 30000 which are classified (positive, negative)

    vectorizer = CountVectorizer(
        analyzer=lambda s: tokenize_tweet(s, spacy_nlp), dtype="uint8"
    )

    classified_tweets = [row.text for row in classified_df.select("text").collect()]

    sentiments_vector = extract_sentiments_vector(classified_df)

    df_countvectorizer = vectorizer.fit_transform(classified_tweets)

    X_train, X_test, y_train, y_test = train_test_split(
        df_countvectorizer, sentiments_vector, test_size=0.2, random_state=0
    )

    classifier = MultinomialNB()
    classifier.fit(X_train, y_train)

    pred = classifier.predict(X_test)

    print(classification_report(y_test, pred))

    print("############ NEW PREDICTIONS ###############")

    unclassified_tweets = [row.text for row in unclassified_df.select("text").collect()]
    # Only predict first 20 unclassified for demonstration purposes
    data = vectorizer.transform(unclassified_tweets[:20])

    pred = classifier.predict(data)

    print(pred)


def extract_sentiments_vector(df):
    return (
        df.select(
            when(df.sentiment1 == "positive", 1).otherwise(0).alias("sentiment_numeric")
        )
        .rdd.map(lambda row: row["sentiment_numeric"])
        .collect()
    )
