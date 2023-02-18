import os
import logging
import numpy as np
import pandas as pd
import category_encoders as ce
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


def fit_transform_item_features(
    item_features: pd.DataFrame,
    count_cols: Optional[list] = None,
    hashing_enc_cols: Optional[list] = None,
    text_emb_col: Optional[str] = None,
    n_factors: Optional[int] = None,
    max_features: Optional[int] = None,
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

    item_features_transformed = pd.concat([df1, df2, df3], axis=1)

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
