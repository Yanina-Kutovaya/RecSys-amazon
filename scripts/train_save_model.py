#!/usr/bin/env python3
"""Train and save model for RecSys-amazon"""

import sys
import os

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), ".."))

import logging
import argparse
import pandas as pd
import lightgbm as lgb
from typing import Optional

from src.recsys_amazon.data.make_dataset import build_dataset
from src.recsys_amazon.data.validation import train_test_split
from src.recsys_amazon.models import train
from src.recsys_amazon.models.serialize import store


logger = logging.getLogger()


def main():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        "-d1",
        "--data_path",
        required=False,
        default="data/01_raw/Grocery_and_Gourmet_Food.csv",
        help="transactions dataset store path",
    )
    argparser.add_argument(
        "-d2",
        "--item_features_path",
        required=False,
        default="data/01_raw/meta_Grocery_and_Gourmet_Food.json.gz",
        help="item features dataset store path",
    )
    argparser.add_argument(
        "-d3",
        "--user_features_path",
        required=False,
        default="data/01_raw/Grocery_and_Gourmet_Food_5.json.gz",
        help="user features dataset store path",
    )
    argparser.add_argument(
        "-o",
        "--output",
        required=True,
        help="filename to store model",
    )
    argparser.add_argument(
        "-v", "--verbose", help="increase output verbosity", action="store_true"
    )
    args = argparser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    logging.info("Reading data...")
    (
        data_train_lvl_1,
        data_val_lvl_1,
        data_val_lvl_2,
        item_features,
        user_reviews,
    ) = build_dataset(args.data_path, args.item_features_path, args.user_features_path)

    logging.info("Preprocessing data...")
    train_dataset_lvl_2 = train.data_preprocessing_pipeline(
        data_train_lvl_1, data_val_lvl_1, data_val_lvl_2, item_features, user_reviews
    )

    logging.info("Training the model...")
    train_store(train_dataset_lvl_2, args.output)


def train_store(dataset: pd.DataFrame, filename: str):
    """
    Trains and stores LightGBM model.
    """

    X_train, X_valid, y_train, y_valid = train_test_split(dataset)
    dtrain = lgb.Dataset(X_train.iloc[:, 2:], y_train)
    dvalid = lgb.Dataset(X_valid.iloc[:, 2:], y_valid)

    logging.info(f"Training the model on {len(X_train)}  items...")

    params_lgb = {
        "boosting_type": "gbdt",
        "objective": "binary",
        "metric": "auc",
        "num_boost_round": 10000,
        "learning_rate": 0.005,
        "num_leaves": 100,
        "max_depth": 15,
        "n_estimators": 5000,
        "n_jobs": 6,
        "seed": 12,
    }

    model_lgb = lgb.train(
        params=params_lgb,
        train_set=dtrain,
        valid_sets=[dtrain, dvalid],
        verbose_eval=1000,
        early_stopping_rounds=30,
    )
    store(model_lgb, filename)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.critical(e)
        sys.exit(1)
