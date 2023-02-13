import logging
import pandas as pd
import json
import gzip
import re
from typing import Optional, Tuple


logger = logging.getLogger(__name__)

__all__ = ["load_items_matadata"]


def get_items_matadata(selected_items: set, item_path: str) -> pd.DataFrame:
    """
    Builds dataset for selected items from metadata
    """

    re_html = re.compile(r"<[^>]+>")
    with gzip.open(item_path) as f:
        items = pd.DataFrame()
        i = 0
        for l in f:
            r = json.loads(l.strip())
            id = r["asin"]
            if id in selected_items:
                items.loc[i, "item_id"] = id
            items.loc[i, "price"] = get_clean_price(r["price"])
            main_cat = r["main_cat"]
            if len(main_cat) < 150:
                items.loc[i, "main_cat"] = main_cat
            items.loc[i, "category_1"] = r["category"][1]
            if len(r["category"]) > 2:
                items.loc[i, "category_2"] = r["category"][2]
            if len(r["category"]) > 3:
                items.loc[i, "category_3"] = r["category"][3]
            items.loc[i, "brand"] = r["brand"]
            rank, rank_group = get_clean_rank(r["rank"])
            items.loc[i, "rank"] = rank
            items.loc[i, "rank_group"] = rank_group
            items.loc[i, "title"] = r["title"]
            text = " ".join(r["description"])
            try:
                items.loc[i, "description"] = re_html.sub("", text)
            except:
                items.loc[i, "description"] = text
            also_view = [i for i in r["also_view"] if i in selected_items]
            items.loc[i, "len_also_view"] = len(also_view)
            items.loc[i, "also_view"] = str(also_view)
            also_buy = [i for i in r["also_buy"] if i in selected_items]
            items.loc[i, "len_also_buy"] = len(also_buy)
            items.loc[i, "also_buy"] = str(also_buy)
            i += 1

    ind = items["item_id"].drop_duplicates().index
    items = items.loc[ind, :]
    items = clean_rank_groups(items)

    return items


def get_clean_price(price):
    if price:
        price = price.replace("$", "").replace(",", "")
        try:
            price = float(price)
        except:
            price = 0
    else:
        price = 0

    return price


def get_clean_rank(rank) -> Tuple[int, str]:
    rank_group = ""
    if not rank:
        rank = 0
    elif type(rank) == str:
        a = rank.split()
        rank_group = " ".join(a[2:-1])
        rank = int(a[0].replace(",", ""))
    elif type(rank) == list:
        a = rank[0].split()
        rank_group = " ".join(a[2:]).split(">")[-1].strip()
        rank = int(a[0][2:].replace(",", ""))
    else:
        rank = 0

    return rank, rank_group


def clean_rank_groups(items: pd.DataFrame) -> pd.DataFrame:
    rank_groups = {}
    rank_groups["Grocery & Gourmet Food"] = [
        "Grocery & Gourmet Food",
        "Grocery & Gourmet Food (See Top 100 in Grocery & Gourmet Food)",
        "Grocery & Gourmet Food (See top 100)",
    ]
    rank_groups["Toys & Games"] = [
        "Toys & Games",
        "Toys & Games (See Top 100 in Toys & Games)",
        "Toys & Games (See top 100)",
    ]
    rank_groups["Industrial & Scientific"] = [
        "Industrial & Scientific",
        "Industrial & Scientific (See Top 100 in Industrial & Scientific)",
        "Industrial & Scientific (See top 100)",
    ]
    rank_groups["Office Products"] = [
        "Office Products",
        "Office Products (See Top 100 in Office Products)",
        "Office Products (See top 100)",
    ]
    rank_groups["Tools & Home Improvement"] = [
        "Tools & Home Improvement",
        "Tools & Home Improvement (See top 100)",
    ]
    for key in rank_groups:
        items.loc[items["rank_group"].isin(rank_groups[key]), "rank_group"] = key

    return items
