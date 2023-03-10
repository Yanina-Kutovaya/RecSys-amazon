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
    Recommendation for new users - N top popular items from the 2nd level model train dataset.

    """

    if n_items is None:
        n_items = N_ITEMS

    data_lvl_2 = pd.concat([data_val_lvl_1, data_val_lvl_2], axis=0)

    user_ids = list(set(data_lvl_2["user_id"]) & set(data_train_lvl_1["user_id"]))
    item_ids = list(set(data_lvl_2["item_id"]) & set(data_train_lvl_1["item_id"]))

    new_users = list(set(data_lvl_2["user_id"]) - set(data_train_lvl_1["user_id"]))

    user_item_lists = [user_ids, item_ids, [0]]
    inference_set = []
    for element in itertools.product(*user_item_lists):
        (user_id, item_id, rating) = element
        inference_set.append([user_id, item_id, rating])

    predictions = recommender.test(inference_set)
    preds = []
    for i in range(len(predictions)):
        item = predictions[i]
        preds.append([item.uid, item.iid, item.est])
    preds = pd.DataFrame(preds, columns=["user_id", "item_id", "rating"])
    preds.sort_values(by=["user_id", "rating"], ascending=False, inplace=True)
    preds = preds.groupby("user_id").head(n_items)[["user_id", "item_id"]]

    top_popular = get_top_popular(data_val_lvl_1, n_items)

    new_user_candidates = get_new_user_candidates(new_users, top_popular)
    candidates_lvl_2 = pd.concat([preds, new_user_candidates], axis=0)

    return candidates_lvl_2


def get_top_popular(
    data_val_lvl_1: pd.DataFrame,
    n_items: int,
) -> list:
    """
    Generates a list of N top popular items based on the 2nd level model train dataset.
    """
    popularity = (
        data_val_lvl_1.groupby("item_id")["user_id"].nunique()
        / data_val_lvl_1["user_id"].nunique()
    ).reset_index()
    popularity.rename(columns={"user_id": "share_unique_users"}, inplace=True)

    top_popular = (
        popularity.sort_values("share_unique_users", ascending=False)
        .iloc[:n_items, 0]
        .tolist()
    )

    return top_popular


def get_new_user_candidates(
    new_users: list,
    top_popular: list,
) -> pd.DataFrame:
    """
    Recommends items from top popular list to new users.

    """
    new_user_item_lists = [new_users, top_popular]
    new_user_set = []
    for element in itertools.product(*new_user_item_lists):
        (user_id, item_id) = element
        new_user_set.append([user_id, item_id])
    new_user_candidates = pd.DataFrame(new_user_set, columns=["user_id", "item_id"])

    return new_user_candidates
