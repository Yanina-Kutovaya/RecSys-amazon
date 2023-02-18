import logging
import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)

__all__ = ["generate_user_item_features"]


def get_user_item_features(
    data_val_lvl_1: pd.DataFrame,
    item_features: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generates new features from users and items data of the 2nd level model train dataset.
    """

    df = data_val_lvl_1.merge(item_features, on="item_id", how="left")
    f = lambda x: x.nunique()
    user_stat = df.groupby("user_id", as_index=False).agg(
        mean_rating_user=("rating", "mean"),
        n_rated_user=("rating", "count"),
        mean_price_user=("price", "mean"),
        total_spent_user=("price", "sum"),
    )
    item_stat = df.groupby("item_id", as_index=False).agg(
        n_rated_item=("rating", "count"),
        total_spent_item=("price", "sum"),
    )
    brand_stat = df.groupby("brand", as_index=False).agg(
        n_users_brand=("user_id", f),
        n_ratings_brand=("rating", "count"),
        mean_rating_brand=("rating", "mean"),
        mean_price_brand=("price", "mean"),
        total_spent_brand=("price", "sum"),
        n_main_categories_brand=("main_cat", f),
        n_category_1_brand=("category_1", f),
        n_category_2_brand=("category_2", f),
        n_category_3_brand=("category_3", f),
        n_rank_groups_brand=("rank_group", f),
    )
    category_2_stat = df.groupby("category_2", as_index=False).agg(
        n_users_category_2=("user_id", f),
        n_ratings_category_2=("rating", "count"),
        mean_rating_category_2=("rating", "mean"),
        mean_price_category_2=("price", "mean"),
        total_spent_category_2=("price", "sum"),
        n_brands_category_2=("brand", f),
    )
    category_3_stat = df.groupby("category_3", as_index=False).agg(
        n_users_category_3=("user_id", f),
        n_ratings_category_3=("rating", "count"),
        mean_rating_category_3=("rating", "mean"),
        mean_price_category_3=("price", "mean"),
        total_spent_category_3=("price", "sum"),
        n_brands_category_3=("brand", f),
    )
    cols = ["user_id", "item_id", "brand", "category_2", "category_3"]
    user_item_features = (
        df[cols]
        .merge(user_stat, on="user_id", how="left")
        .merge(item_stat, on="item_id", how="left")
        .merge(brand_stat, on="brand", how="left")
        .merge(category_2_stat, on="category_2", how="left")
        .merge(category_3_stat, on="category_3", how="left")
        .drop(["brand", "category_2", "category_3"], axis=1)
        .fillna(0)
    )

    return user_item_features
