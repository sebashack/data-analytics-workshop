from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaMulticore
from gensim.models import CoherenceModel

from tf_idf import (
    tokenize_dataset,
)


def topic_modelling_pipeline(
    spacy_nlp, df, no_below, no_above, keep_n, workers, passes, num_topics
):
    tokenized_df = tokenize_dataset(df, spacy_nlp)
    tokens = tokenized_tweets_to_list(tokenized_df)
    dictionary = get_dictionary_from_tokens(tokens, no_below, no_above, keep_n)
    corpus = [dictionary.doc2bow(doc) for doc in tokens]
    lda_model = compute_topic_model(corpus, dictionary, num_topics, workers, passes)
    coherence_u_mass = get_coherence(lda_model, corpus, dictionary, "u_mass", tokens)
    coherence_c_v = get_coherence(lda_model, corpus, dictionary, "c_v", tokens)
    print("Coherence_u_mass: ", coherence_u_mass)
    print("Coherence_c_v: ", coherence_c_v)

    return lda_model


def get_dictionary_from_tokens(tokens, no_below, no_above, keep_n):
    dictionary = Dictionary(tokens)
    dictionary.filter_extremes(no_below=no_below, no_above=no_above, keep_n=keep_n)

    return dictionary


def tokenized_tweets_to_list(tokenized_df):
    return [row["tokenized_text"] for row in tokenized_df.collect()]


def compute_topic_model(corpus, dictionary, num_topics, workers, passes):
    return LdaMulticore(
        corpus=corpus,
        id2word=dictionary,
        iterations=10,
        num_topics=num_topics,
        workers=workers,
        passes=passes,
    )


def get_model_lda_and_k_optimal(
    spacy_nlp,
    df,
    no_below,
    no_above,
    keep_n,
    workers,
    passes,
    max_topics,
    type_coherence,
):
    tokenized_df = tokenize_dataset(df, spacy_nlp)
    tokens = tokenized_tweets_to_list(tokenized_df)
    dictionary = get_dictionary_from_tokens(tokens, no_below, no_above, keep_n)
    corpus = [dictionary.doc2bow(doc) for doc in tokens]

    topics = []
    score = []
    lda_model = None
    for i in range(1, max_topics):
        lda_model = LdaMulticore(
            corpus=corpus,
            id2word=dictionary,
            iterations=10,
            num_topics=i,
            workers=workers,
            passes=passes,
            random_state=100,
        )
        coherence = get_coherence(lda_model, corpus, dictionary, type_coherence, tokens)
        topics.append(i)
        score.append(coherence)

    idx_max_coherence = score.index(max(score))

    return lda_model, topics[idx_max_coherence]


def get_coherence(lda_model, corpus, dictionary, type_coherence, tokens):
    cm = None
    if type_coherence == "u_mass":
        cm = CoherenceModel(
            model=lda_model,
            corpus=corpus,
            dictionary=dictionary,
            coherence=type_coherence,
        )
    else:
        cm = CoherenceModel(
            model=lda_model,
            texts=tokens,
            corpus=corpus,
            dictionary=dictionary,
            coherence="c_v",
        )

    return cm.get_coherence()
