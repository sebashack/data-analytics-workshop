from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from pyspark.sql.functions import when, monotonically_increasing_id
from sklearn.feature_extraction.text import CountVectorizer
from pyspark.sql.functions import udf
from pyspark.sql.types import IntegerType
from tf_idf import (
    tokenize_tweet,
)

HASHTAG_TO_INT = {
    "#bushfires": 0,
    "#climatecrisis": 1,
    "#globalwarming": 2,
    "#sustainability": 3,
    "#climatestrike": 4,
    "#actonclimate": 5,
    "#savetheplanet": 6,
    "#climateaction": 7,
    "#greennewdeal": 8,
    "#environment": 9,
    "#fridaysforfuture": 10,
    "#climatechange": 11,
}


def classification_pipeline(df, spacy_nlp):
    h = get_unique_hashtags(df)

    vectorizer = CountVectorizer(
        analyzer=lambda s: tokenize_tweet(s, spacy_nlp), dtype="uint8"
    )

    tweets = [row.text for row in df.select("text").collect()]

    topic_vector = extract_topic_vector(df)

    df_countvectorizer = vectorizer.fit_transform(tweets)

    X_train, X_test, y_train, y_test = train_test_split(
        df_countvectorizer, topic_vector, test_size=0.2, random_state=0
    )

    classifier = MultinomialNB()
    classifier.fit(X_train, y_train)

    pred = classifier.predict(X_test)

    print(classification_report(y_test, pred))


def extract_topic_vector(df):
    return list(
        map(
            lambda h: HASHTAG_TO_INT[h],
            [row["search_hashtags"] for row in df.collect()],
        )
    )


def get_unique_hashtags(df):
    hashtags_list = [row["search_hashtags"] for row in df.collect()]

    unique_hashtags = set()

    for ht in hashtags_list:
        if ht is not None and ht.startswith("#"):
            unique_hashtags.add(ht)

    return unique_hashtags
