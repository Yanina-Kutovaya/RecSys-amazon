import logging
import numpy as np
import pandas as pd

import nltk
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text as tx
import implicit

from typing import Optional

nltk.download("punkt")


logger = logging.getLogger(__name__)

__all__ = ["transform_user_features"]


MAX_FEATURES = 500
N_FACTORS = 50


def fit_transform_user_features(user_reviews: pd.DataFrame) -> pd.DataFrame:
    """
    1. Calculates the number of words in user review.
    2. Calculates the number of sentences in each user review.
    3. Calculates mean number of characters in sentences of each user review.
    4. Calculates the length of the 1st in sentence of each user review.
    5. Calculates the number of words in the 1st in sentence of each user review.
    6. Generates user reviews embeddings.

    """

    def get_n_words(text):
        return len(text.split())

    user_reviews["n_words"] = user_reviews["text"].apply(get_n_words)
    user_reviews["text_"] = user_reviews["text"].apply(sent_tokenize)

    def get_n_sentence(sent_list):
        return len(sent_list)

    user_reviews["n_sentence"] = user_reviews["text_"].apply(get_n_sentence)
    user_reviews["mean_sent_len"] = (
        user_reviews["len_text"] / user_reviews["n_sentence"]
    )

    def get_len_1st_sentance(text):
        return len(text[0])

    user_reviews["len_1st_sentence"] = user_reviews["text_"].apply(get_len_1st_sentance)

    def get_n_words_1st_sentance(text):
        return len(text[0].split())

    user_reviews["n_words_1st_sentence"] = user_reviews["text_"].apply(
        get_n_words_1st_sentance
    )

    user_embeddings = get_text_embeddings(user_reviews)

    cols = [
        "user_id",
        "item_id",
        "len_text",
        "n_words",
        "n_sentence",
        "mean_sent_len",
        "len_1st_sentence",
        "n_words_1st_sentence",
    ]
    user_features_transformed = pd.concat([user_reviews[cols], user_embeddings], axis=1)

    return user_features_transformed


def get_text_embeddings(
    data: pd.DataFrame,
    col: Optional[str] = "text",
    max_features: Optional[int] = None,
    n_factors: Optional[int] = None,
    prefix: Optional[str] = "r_",
) -> pd.DataFrame:

    if max_features is None:
        max_features = MAX_FEATURES

    if n_factors is None:
        n_factors = N_FACTORS

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        analyzer="word",
        lowercase=True,
        max_features=max_features,
        stop_words=tx.ENGLISH_STOP_WORDS,
    )
    X = vectorizer.fit_transform(data[col])

    als = implicit.als.AlternatingLeastSquares(
        factors=n_factors,
        iterations=30,
        use_gpu=False,
        calculate_training_loss=False,
        regularization=0.1,
    )
    als.fit(X)
    cols = [prefix + str(i) for i in range(n_factors)]
    df = pd.DataFrame(als.user_factors, columns=cols)

    return df
