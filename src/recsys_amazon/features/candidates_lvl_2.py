import logging
import pandas as pd
import itertools
from typing import Optional, Tuple


logger = logging.getLogger(__name__)

__all__ = ["getting_candidates_for_level_2"]

N_ITEMS = 100


def get_candidates(
    recommender,
    data_train_lvl_1: pd.DataFrame,
    data_val_lvl_1: pd.DataFrame,
    data_val_lvl_2: pd.DataFrame,
    n_items: Optional[int] = None,
) -> pd.DataFrame:

    """
    Generates candidates for the 2nd level model.
    Each user from train dataset of the 1st level model is recommended N items.
    Recommendation for new users - N top popular items from train and validation
    data of the 1st level model.

    """

    if n_items is None:
        n_items = N_ITEMS

    user_ids = data_train_lvl_1["user_id"].unique().tolist()
    item_ids = data_train_lvl_1["item_id"].unique().tolist()

    user_item_lists = [user_ids, item_ids, [0]]
    inference_set = []
    for element in itertools.product(*user_item_lists):
        (user_id, item_id, rating) = element
        inference_set.append([user_id, item_id, rating])

    predictions = recommender.test(inference_set)
    preds = pd.DataFrame(columns=["user_id", "item_id", "rating"])
    for i in range(len(predictions)):
        item = predictions[i]
        preds.loc[i, "user_id"] = item.uid
        preds.loc[i, "item_id"] = item.iid
        preds.loc[i, "rating"] = item.est

    preds.sort_values(by=["user_id", "rating"], ascending=False, inplace=True)
    preds = preds.groupby("user_id").head(n_items)[["user_id", "item_id"]]

    top_popular = get_top_popular(data_train_lvl_1, data_val_lvl_1, n_items)
    new_user_candidates = get_new_user_candidates(
        data_train_lvl_1, data_val_lvl_1, data_val_lvl_2, top_popular
    )
    candidates_lvl_2 = pd.concat([preds, new_user_candidates], axis=0)

    return candidates_lvl_2


def get_top_popular(
    data_train_lvl_1: pd.DataFrame,
    data_val_lvl_1: pd.DataFrame,
    n_items: int,
) -> list:
    """
    Generates a list of N top popular items based on train and validation data from the 1st level model
    """
    data = pd.concat([data_train_lvl_1, data_val_lvl_1], axis=0)
    popularity = (
        data.groupby("item_id")["user_id"].nunique() / data["user_id"].nunique()
    ).reset_index()
    popularity.rename(columns={"user_id": "share_unique_users"}, inplace=True)

    top_popular = (
        popularity.sort_values("share_unique_users", ascending=False)
        .iloc[:n_items, 0]
        .tolist()
    )

    return top_popular


def get_new_user_candidates(
    data_train_lvl_1: pd.DataFrame,
    data_val_lvl_1: pd.DataFrame,
    data_val_lvl_2: pd.DataFrame,
    top_popular: list,
) -> pd.DataFrame:
    """
    Recommends to new users items from top popular list.

    """
    all_users = pd.concat([data_train_lvl_1, data_val_lvl_1, data_val_lvl_2], axis=0)[
        "user_id"
    ].unique()
    user_ids = data_train_lvl_1["user_id"].unique()
    new_users = list(set(all_users) - set(user_ids))
    new_user_item_lists = [new_users, top_popular]
    new_user_set = []
    for element in itertools.product(*new_user_item_lists):
        (user_id, item_id) = element
        new_user_set.append([user_id, item_id])
    new_user_candidates = pd.DataFrame(new_user_set, columns=["user_id", "item_id"])

    return new_user_candidates
