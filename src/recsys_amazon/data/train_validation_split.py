import logging
import pandas as pd
from typing import Tuple


logger = logging.getLogger(__name__)

__all__ = ["train_validation_split"]


def time_split(
    t0: int, t1: int, t2: int, user_ids: set, item_ids: set, data: pd.DataFrame
) -> Tuple[set, set, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Train - validation - test schema:
    -- 5 years -- | -- 2 years -- | -- 1 year --

    For the 1st level model we use 5-years data for train and 2-years data for validation.
    For the 2nd level mmodel we use 2-years data for train and 1-year - for validation.
    Validation data for the 1st levl model serves as train data for the 2nd level model.

    t0, t1 - start and end of 5-year train period for the 1st level model
    t1, t2 - start and end of 2-year validation period for the 1st level model
           - start and end of 2-year train period for the 2nd level model
    t2 - start of 1-year validation period for the 2nd level model

    user_ids - users who rated at least 3 items
    item_ids - items rated by at least 2 useres

    """

    df_1_y = data[
        data["user_id"].isin(user_ids)
        & data["item_id"].isin(item_ids)
        & (data["timestamp"] >= t2)
    ].drop_duplicates()
    df_2_y = data[
        data["user_id"].isin(user_ids)
        & data["item_id"].isin(item_ids)
        & (data["timestamp"] >= t1)
        & (data["timestamp"] < t2)
    ].drop_duplicates()
    selected_uses = set(df_2_y["user_id"]) - (
        set(df_1_y["user_id"]) - set(df_2_y["user_id"])
    )
    selected_items = set(df_2_y["item_id"]) | set(df_1_y["item_id"])

    df_5_y = data[
        data["user_id"].isin(selected_uses)
        & data["item_id"].isin(selected_items)
        & (data["timestamp"] >= t0)
        & (data["timestamp"] < t1)
    ].drop_duplicates()

    return (
        selected_uses,
        selected_items,
        df_5_y,
        df_2_y,
        df_1_y,
    )
