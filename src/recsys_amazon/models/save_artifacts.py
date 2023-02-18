import os
import logging
import joblib
import pandas as pd
from typing import Optional


__all__ = ["save_artifacts"]

logger = logging.getLogger()

PATH = ""

FOLDER_2 = "data/02_intermediate/"
TRAIN_DATA_LEVEL_1_PATH = FOLDER_2 + "data_train_lvl_1.parquet.gzip"
VALID_DATA_LEVEL_2_PATH = FOLDER_2 + "data_val_lvl_2.parquet.gzip"
ITEM_FEATURES_PATH = FOLDER_2 + "item_features.parquet.gzip"
USER_REVIEWS_PATH = FOLDER_2 + "user_reviews.parquet.gzip"

FOLDER_3 = "data/03_primary/"
CANDIDATES_PATH = FOLDER_3 + "candidates_lvl_2.parquet.gzip"

FOLDER_4 = "data/04_feature/"
PREFILTERED_ITEM_LIST_PATH = FOLDER_4 + "prefiltered_item_list.joblib"
CURRENT_USER_LIST_PATH = FOLDER_4 + "current_user_list.joblib"
VALID_DATA_LEVEL_1_PATH = FOLDER_4 + "data_val_lvl_1.parquet.gzip"
RECOMMENDER_PATH = FOLDER_4 + "recommender_v1.joblib"
ITEM_FEATURES_TRANSFORMED_PATH = FOLDER_4 + "item_features_transformed.parquet.gzip"
USER_FEATURES_TRANSFORMED_PATH = FOLDER_4 + "user_features_transformed.parquet.gzip"
USER_ITEM_FEATURES_PATH = FOLDER_4 + "user_item_features.parquet.gzip"

FOLDER_5 = "data/05_model_input/"
TRAIN_DATASET_LVL_2_PATH = FOLDER_5 + "train_dataset_lvl_2.parquet.gzip"

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")


def save_dataset(
    data_train_lvl_1: pd.DataFrame,
    data_val_lvl_1: pd.DataFrame,
    data_val_lvl_2: pd.DataFrame,
    item_features: pd.DataFrame,
    user_reviews: pd.DataFrame,
    path: Optional[str] = None,
    train_data_lvl_1_path: Optional[str] = None,
    valid_data_level_1_path: Optional[str] = None,
    valid_data_level_2_path: Optional[str] = None,
    item_features_path: Optional[str] = None,
    user_reviews_path: Optional[str] = None,
):

    logging.info("Saving dataset...")

    if path is None:
        path = PATH

    if train_data_lvl_1_path is None:
        train_data_lvl_1_path = path + TRAIN_DATA_LEVEL_1_PATH
    data_train_lvl_1.to_parquet(train_data_lvl_1_path, compression="gzip", index=False)

    if valid_data_level_1_path is None:
        valid_data_level_1_path = path + VALID_DATA_LEVEL_1_PATH
    data_val_lvl_1.to_parquet(valid_data_level_1_path, compression="gzip", index=False)

    if valid_data_level_2_path is None:
        valid_data_level_2_path = path + VALID_DATA_LEVEL_2_PATH
    data_val_lvl_2.to_parquet(valid_data_level_2_path, compression="gzip", index=False)

    if item_features_path is None:
        item_features_path = path + ITEM_FEATURES_PATH
    item_features.to_parquet(item_features_path, compression="gzip")

    if user_reviews_path is None:
        user_reviews_path = path + USER_REVIEWS_PATH
    user_reviews.to_parquet(user_reviews_path, compression="gzip")


def load_dataset(
    path: Optional[str] = None,
    train_data_lvl_1_path: Optional[str] = None,
    valid_data_level_1_path: Optional[str] = None,
    valid_data_level_2_path: Optional[str] = None,
    item_features_path: Optional[str] = None,
    user_reviews_path: Optional[str] = None,
):
    logging.info("Loading dataset...")

    if path is None:
        path = PATH

    if train_data_lvl_1_path is None:
        train_data_lvl_1_path = path + TRAIN_DATA_LEVEL_1_PATH
    data_train_lvl_1 = pd.read_parquet(train_data_lvl_1_path)

    if valid_data_level_1_path is None:
        valid_data_level_1_path = path + VALID_DATA_LEVEL_1_PATH
    data_val_lvl_1 = pd.read_parquet(valid_data_level_1_path)

    if valid_data_level_2_path is None:
        valid_data_level_2_path = path + VALID_DATA_LEVEL_2_PATH
    data_val_lvl_2 = pd.read_parquet(valid_data_level_2_path)

    if item_features_path is None:
        item_features_path = path + ITEM_FEATURES_PATH
    item_features = pd.read_parquet(item_features_path)

    if user_reviews_path is None:
        user_reviews_path = path + USER_REVIEWS_PATH
    user_reviews = pd.read_parquet(user_reviews_path)

    return data_train_lvl_1, data_val_lvl_1, data_val_lvl_2, item_features, user_reviews


def save_prefiltered_item_list(
    prefiltered_item_list: list,
    path: Optional[str] = None,
    prefiltered_item_list_path: Optional[str] = None,
):

    logging.info("Saving prefiltered item list...")

    if path is None:
        path = PATH

    if prefiltered_item_list_path is None:
        prefiltered_item_list_path = path + PREFILTERED_ITEM_LIST_PATH
    joblib.dump(prefiltered_item_list, prefiltered_item_list_path)


def save_current_user_list(
    current_user_list: list,
    path: Optional[str] = None,
    current_user_list_path: Optional[str] = None,
):

    logging.info("Saving current user list...")

    if path is None:
        path = PATH

    if current_user_list_path is None:
        current_user_list_path = path + CURRENT_USER_LIST_PATH

    joblib.dump(current_user_list, current_user_list_path, 3)


def save_recommender(
    recommender, path: Optional[str] = None, recommender_path: Optional[str] = None
):

    logging.info("Saving recommender...")

    if path is None:
        path = PATH

    if recommender_path is None:
        recommender_path = path + RECOMMENDER_PATH
    joblib.dump(recommender, recommender_path, 3)


def save_candidates(
    users_lvl_2: pd.DataFrame,
    path: Optional[str] = None,
    candidates_path: Optional[str] = None,
):

    logging.info("Saving candidates for level 2 model...")

    if path is None:
        path = PATH

    if candidates_path is None:
        candidates_path = path + CANDIDATES_PATH
    users_lvl_2.to_parquet(candidates_path, compression="gzip")


def save_item_featutes(
    item_features_transformed: pd.DataFrame,
    path: Optional[str] = None,
    item_features_transformed_path: Optional[str] = None,
):

    logging.info("Saving item features...")

    if path is None:
        path = PATH

    if item_features_transformed_path is None:
        item_features_transformed_path = path + ITEM_FEATURES_TRANSFORMED_PATH
    item_features_transformed.to_parquet(
        item_features_transformed_path, compression="gzip"
    )


def save_user_features(
    user_features_transformed: pd.DataFrame,
    path: Optional[str] = None,
    user_features_transformed_path: Optional[str] = None,
):

    logging.info("Saving user features...")

    if path is None:
        path = PATH

    if user_features_transformed_path is None:
        user_features_transformed_path = path + USER_FEATURES_TRANSFORMED_PATH
    user_features_transformed.to_parquet(
        user_features_transformed_path, compression="gzip"
    )


def save_user_item_features(
    user_item_features: pd.DataFrame,
    path: Optional[str] = None,
    user_item_features_path: Optional[str] = None,
):

    logging.info("Saving new user-item features...")

    if path is None:
        path = PATH

    if user_item_features_path is None:
        user_item_features_path = path + USER_ITEM_FEATURES_PATH
    user_item_features.to_parquet(user_item_features_path, compression="gzip")


def save_train_dataset_lvl_2(
    train_dataset_lvl_2: pd.DataFrame,
    path: Optional[str] = None,
    train_dataset_lvl_2_path: Optional[str] = None,
):

    logging.info("Saving train dataset for level 2 model...")

    if path is None:
        path = PATH

    if train_dataset_lvl_2_path is None:
        train_dataset_lvl_2_path = path + TRAIN_DATASET_LVL_2_PATH
    train_dataset_lvl_2.to_parquet(train_dataset_lvl_2_path, compression="gzip")
