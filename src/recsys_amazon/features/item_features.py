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
FEATURE_FOR_HASHING_ENCODER = "brand"
MAX_FEATURES = 500
N_FACTORS = 50


def fit_transform_item_features(
    item_features: pd.DataFrame,
    count_cols=None,
    hashing_enc_col=None,
) -> pd.DataFrame:
    """
    Generates new item featurs for train dataset.
    """
    logging.info("Transforming item_features for train dataset...")

    if count_cols is None:
        count_cols = FEATURES_FOR_COUNT_ENCODER
    if hashing_enc_col is None:
        hashing_enc_col = FEATURE_FOR_HASHING_ENCODER

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

    hashing_encoder = ce.HashingEncoder(cols=[hashing_enc_col])
    df2 = hashing_encoder.fit_transform(item_features[[hashing_enc_col]])
    df2.columns = [hashing_enc_col + "_" + str(i) for i in range(df2.shape[1])]

    logging.info("Encoding item descriptions...")

    df3 = get_text_embeddings(
        item_features,
        col="description",
        n_factors=N_FACTORS,
        max_features=MAX_FEATURES,
        prefix="d_",
    )

    item_features_transformed = pd.concat([df1, df2, df3], axis=1)

    return item_features_transformed
