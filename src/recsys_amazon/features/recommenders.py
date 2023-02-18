import logging
import pandas as pd
from surprise import SVD, Dataset, Reader
from typing import Optional

logger = logging.getLogger(__name__)

__all__ = ["getting_candidates_for_level_2"]


DEFAULT_RANDOM_SEED = 25


def get_recommender(data_train_lvl_1: pd.DataFrame, seed: Optional[int] = None):
    """
    Generates recommender based on surprise library
    """

    if seed is None:
        seed = DEFAULT_RANDOM_SEED

    reader = Reader(
        line_format="user item rating",
        rating_scale=(1, 5),
    )
    trainset = Dataset.load_from_df(
        data_train_lvl_1[["user_id", "item_id", "rating"]], reader
    ).build_full_trainset()

    recommender = SVD(random_state=seed)
    recommender.fit(trainset)

    return recommender
