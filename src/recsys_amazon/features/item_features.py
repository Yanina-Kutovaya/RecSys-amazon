import os
import logging
import numpy as np
import pandas as pd
import category_encoders as ce
import implicit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text
from typing import Optional

from .user_features import get_text_embeddings


logger = logging.getLogger(__name__)

__all__ = ["transform_item_features"]


FEATURES_FOR_COUNT_ENCODER = [
    "main_cat",
    "category_1",
    "category_2",
    "brand",
    "rank_group",
]
FEATURES_FOR_HASHING_ENCODER = ["brand", "rank_group"]

FEATURE_FOR_TEXT_EMBEDDINGS = "description"
MAX_FEATURES = 500
N_FACTORS = 50

FEATURES_WITH_ITEMS_LISTS = ["also_view", "also_buy"]
N_FACTORS_ITEMS_LISTS = 50


def fit_transform_item_features(
    item_features: pd.DataFrame,
    count_cols: Optional[list] = None,
    hashing_enc_cols: Optional[list] = None,
    text_emb_col: Optional[str] = None,
    n_factors: Optional[int] = None,
    max_features: Optional[int] = None,
    items_lists_cols: Optional[list] = None,
) -> pd.DataFrame:
    """
    Generates new item featurs for train dataset.
    """
    logging.info("Transforming item_features for train dataset...")

    if count_cols is None:
        count_cols = FEATURES_FOR_COUNT_ENCODER
    if hashing_enc_cols is None:
        hashing_enc_cols = FEATURES_FOR_HASHING_ENCODER
    if text_emb_col is None:
        text_emb_col = FEATURE_FOR_TEXT_EMBEDDINGS
    if n_factors is None:
        n_factors = N_FACTORS
    if max_features is None:
        max_features = MAX_FEATURES
    if items_lists_cols is None:
        items_lists_cols = FEATURES_WITH_ITEMS_LISTS

    item_features.set_index("item_id", inplace=True)

    logging.info("Encoding item features with CountEncoder ...")

    count_encoder = ce.CountEncoder(
        cols=count_cols,
        handle_unknown=-1,
        handle_missing=-2,
        min_group_size=5,
        combine_min_nan_groups=True,
        min_group_name="others",
        normalize=True,
    )
    df1 = count_encoder.fit_transform(item_features[count_cols])
    df1.columns = [i + "_count" for i in df1.columns]

    logging.info("Encoding features with HashingEncoder ...")

    df2 = pd.DataFrame(index=item_features.index)
    for feature in hashing_enc_cols:
        df = hashing_item_features(item_features, feature)
        df2 = pd.concat([df2, df], axis=1)

    logging.info("Encoding item descriptions...")

    df3 = get_text_embeddings(
        item_features,
        col=text_emb_col,
        n_factors=n_factors,
        max_features=max_features,
        prefix="d_",
    )
    df3.index = item_features.index

    logging.info("Encoding features with items lists...")

    df4 = pd.DataFrame(index=item_features.index)
    for i, feature in enumerate(items_lists_cols):
        prefix = "s" + str(i) + "_"
        df = get_items_lists_embeddngs(item_features, feature, prefix)
        df4 = pd.concat([df4, df], axis=1)

    num_cols = item_features.dtypes[item_features.dtypes == "float"].keys()
    item_features_transformed = pd.concat(
        [item_features[num_cols], df1, df2, df3, df4], axis=1
    )

    return item_features_transformed.reset_index()


def hashing_item_features(
    item_features: pd.DataFrame,
    feature: str,
) -> pd.DataFrame:
    hashing_encoder = ce.HashingEncoder(cols=[feature])
    df = hashing_encoder.fit_transform(item_features[[feature]])
    df.columns = [feature + "_" + str(i) for i in range(df.shape[1])]
    df.index = item_features.index

    return df


def get_items_lists_embeddngs(
    item_features: pd.DataFrame,
    col: str,
    prefix: str,
    n_factors: Optional[int] = None,
) -> pd.DataFrame:

    if n_factors is None:
        n_factors = N_FACTORS_ITEMS_LISTS

    vectorizer = TfidfVectorizer(lowercase=False)
    X = vectorizer.fit_transform(item_features[col])

    als = implicit.als.AlternatingLeastSquares(
        factors=n_factors,
        iterations=30,
        use_gpu=False,
        calculate_training_loss=False,
        regularization=0.1,
    )
    als.fit(X)

    cols = [prefix + "1_" + str(i) for i in range(n_factors)]
    df1 = pd.DataFrame(als.user_factors, index=item_features.index, columns=cols)

    item_index = vectorizer.get_feature_names_out().tolist()
    cols = [prefix + "2_" + str(i) for i in range(n_factors)]
    df2 = pd.DataFrame(als.item_factors, index=item_index, columns=cols)

    return pd.concat([df1, df2], axis=1)
