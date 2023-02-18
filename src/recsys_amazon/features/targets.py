import logging
import numpy as np
import pandas as pd
from typing import Optional

logger = logging.getLogger(__name__)

__all__ = ["generate_targets"]


def get_targets(
    data_val_lvl_1: pd.DataFrame,
    candidates_lvl_2: pd.DataFrame,
    item_features_transformed: pd.DataFrame,
    user_features_transformed: pd.DataFrame,
    user_item_features: pd.DataFrame,
) -> pd.DataFrame:
    """
    Selects only top rated items in the 2nd level model train dataset and rates them as 1.
    The rest of the items are assumeed to be rated by users as 0.
    Adds transfored users and items features to the 2nd level model train dataset.

    """
    targets_lvl_2 = data_val_lvl_1.loc[
        data_val_lvl_1["rating"] == 5, ["user_id", "item_id"]
    ]
    targets_lvl_2["target"] = 1
    targets_lvl_2 = (
        candidates_lvl_2.merge(targets_lvl_2, on=["user_id", "item_id"], how="outer")
        .fillna(0)
        .drop_duplicates()
        .reset_index(drop=True)
    )
    targets_lvl_2 = targets_lvl_2.merge(
        item_features_transformed, on="item_id", how="left"
    )
    targets_lvl_2 = targets_lvl_2.merge(
        user_features_transformed, on="user_id", how="left"
    )
    targets_lvl_2 = targets_lvl_2.merge(
        user_item_features, on=["user_id", "item_id"], how="left"
    ).drop_duplicates()

    return targets_lvl_2
