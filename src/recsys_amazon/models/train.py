import sys
import os

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "src", "recsys_amazon"))

import logging
import numpy as np
import pandas as pd
from pickle import dump
from typing import Optional

from data.make_dataset import build_dataset
from features.recommenders import get_recommender
from features.candidates_lvl_2 import get_candidates
from features.user_features import fit_transform_user_features, get_text_embeddings
from features.item_features import fit_transform_item_features
from features.new_item_user_features import get_user_item_features
from features.targets import get_targets_lvl_2
from .save_artifacts import (
    save_dataset,
    save_recommender,
    save_candidates,
    save_item_featutes,
    save_user_features,
    save_user_item_features,
    save_train_dataset_lvl_2,
)

logger = logging.getLogger(__name__)

__all__ = ["preprocess_data"]


N_ITEMS = 100

PATH = "data/"
FOLDERS = ["02_intermediate/", "03_primary/", "04_feature/", "05_model_input/"]


def data_preprocessing_pipeline(
    data_train_lvl_1: pd.DataFrame,
    data_val_lvl_1: pd.DataFrame,
    data_val_lvl_2: pd.DataFrame,
    item_features: pd.DataFrame,
    user_reviews: pd.DataFrame,
    n_factors_ALS: Optional[int] = None,
    n_items: Optional[int] = None,
    save_artifacts=True,
) -> pd.DataFrame:

    """
    Prepares dataset from ratings only data, item features and user features
    to be used in binary classification models.
    n_items - the number of items selected by the recommender for each user
              on the 1st stage (long list), the bases for the further short list
              selection with the binary classification model.
    """

    logging.info("Training recommender...")

    recommender = get_recommender(data_train_lvl_1)

    logging.info("Selecting candidates for level 2 dataset...")

    if n_items is None:
        n_items = N_ITEMS

    candidates_lvl_2 = get_candidates(
        recommender, data_train_lvl_1, data_val_lvl_1, data_val_lvl_2, n_items
    )

    logging.info("Generating new features for level 2 model...")

    item_features_transformed = fit_transform_item_features(item_features)
    user_features_transformed = fit_transform_user_features(user_reviews)
    user_item_features = get_user_item_features(data_val_lvl_1, item_features)

    logging.info("Generating train dataset for level 2 model...")

    train_dataset_lvl_2 = get_targets_lvl_2(
        data_val_lvl_1,
        candidates_lvl_2,
        item_features_transformed,
        user_features_transformed,
        user_item_features,
    )

    if save_artifacts:
        save_dataset(
            data_train_lvl_1,
            data_val_lvl_1,
            data_val_lvl_2,
            item_features,
            user_reviews,
        )
        save_recommender(recommender)
        save_candidates(candidates_lvl_2)
        save_item_featutes(item_features_transformed)
        save_user_features(user_features_transformed)
        save_user_item_features(user_item_features)
        save_train_dataset_lvl_2(train_dataset_lvl_2)

    return train_dataset_lvl_2
