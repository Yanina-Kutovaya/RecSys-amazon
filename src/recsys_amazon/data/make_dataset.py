import logging
import pandas as pd
from typing import Optional, Tuple

from .train_validation_split import time_split
from .item_data import get_items_matadata
from .user_data import get_user_reviews
from src.recsys_amazon.models.save_artifacts import (
    save_prefiltered_item_list,
    save_current_user_list,
)

logger = logging.getLogger(__name__)

__all__ = ["load_train_dataset"]

PATH = "data/01_raw/"
DATA_PATH = PATH + "Grocery_and_Gourmet_Food.csv"
ITEM_FEATURES_PATH = PATH + "meta_Grocery_and_Gourmet_Food.json.gz"
USER_FEATURES_PATH = PATH + "Grocery_and_Gourmet_Food_5.json.gz"


def build_dataset(
    data_path: Optional[str] = None,
    item_path: Optional[str] = None,
    user_path: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Input data source https://nijianmo.github.io/amazon/index.html

    1. From  "Grocery_and_Gourmet_Food.csv" (ratings only) selects users who rated
    at least 3 items and items rated by at least 2 useres.

    2. Splits ratings into train and validation datasets for two-stage recommender system:

    Train - validation - test schema:
    -- 5 years -- | -- 2 years -- | -- 1 year --

    For the 1st level model we use 5-years data for train and 2-years data for validation.
    For the 2nd level mmodel we use 2-years data for train and 1-year - for validation.
    Validation data for the 1st levl model serves as train data for the 2nd level model.

    3. From "meta_Grocery_and_Gourmet_Food.json.gz" (matadata), for items selected for the 2nd
    level model, builds items matadata dataset.

    4. From "Grocery_and_Gourmet_Food_5.json.gz" (5-core), for items selected for the 2nd level
    model, builds user reviews datasets.

    Output:
    data_train_lvl_1 - train dataset with selected users and items for the 1st level model (ratings only)
    data_val_lvl_1 - validation dataset with selected users and items for the 1st level model and
                    rain dataset for the 2nd level model (ratings with users reviews).
    data_val_lvl_2 - validation dataset with selected users and items for the 2nd level model
                    (ratings with users reviews)
    item_features - items matadata dataset built for selected items.
    user_reviews - user reviews dataset built for selected items.

    """

    if data_path is None:
        data_path = DATA_PATH

    logging.info(f"Reading dataset from {data_path}...")
    data = load_ratings(data_path)

    logging.info(f"Selecting user-item pairs...")
    user_ids, item_ids = select_users_item_pairs(data)

    logging.info(f"Splitting data into train-validation datasets...")
    year = 365 * 24 * 3600
    t2 = data["timestamp"].max() - year
    t1 = t2 - 2 * year
    t0 = t1 - 5 * year
    (
        selected_users,
        selected_items,
        data_train_lvl_1,
        data_val_lvl_1,
        data_val_lvl_2,
    ) = time_split(t0, t1, t2, user_ids, item_ids, data)
    save_current_user_list(list(selected_users))
    save_prefiltered_item_list(list(selected_items))

    if item_path is None:
        item_path = ITEM_FEATURES_PATH
    logging.info(f"Reading item_features from {item_path}...")
    item_features = get_items_matadata(selected_items, item_path)

    if user_path is None:
        user_path = USER_FEATURES_PATH
    logging.info(f"Reading user reviews from {user_path}...")
    user_reviews = get_user_reviews(selected_users, selected_items, t1, user_path)

    return data_train_lvl_1, data_val_lvl_1, data_val_lvl_2, item_features, user_reviews


def load_ratings(data_path: str) -> pd.DataFrame:
    data = pd.read_csv(data_path, header=None)
    data.rename(
        columns={0: "item_id", 1: "user_id", 2: "rating", 3: "timestamp"}, inplace=True
    )

    return data


def select_users_item_pairs(
    data: pd.DataFrame, n_items=3, n_users=2
) -> Tuple[set, set]:
    """
    Selects users who rated at least 3 items and items rated by at least 2 useres.
    """
    df = data.groupby(["user_id", "item_id"])["item_id"].count()
    user_ids = set([i for (i, _) in df[df >= n_items].index])
    item_ids = set([i for (_, i) in df[df >= n_users].index])

    return user_ids, item_ids
