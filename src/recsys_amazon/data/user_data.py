import logging
import pandas as pd
import json
import gzip
import re


logger = logging.getLogger(__name__)

__all__ = ["load_user_reviews"]


def get_user_reviews(
    selected_uses: set, selected_items: set, t1: int, user_path: str
) -> pd.DataFrame:
    """
    Builds users reviews from 5-core dataset with selected users and items.
    """

    re_html = re.compile(r"<[^>]+>")
    reviews = []
    with gzip.open(user_path) as f:
        for l in f:
            r = json.loads(l)
            time = r["unixReviewTime"]
            item_id = r["asin"]
            user_id = r["reviewerID"]
            if time >= t1 and item_id in selected_items and user_id in selected_uses:
                review = {}
                review["user_id"] = user_id
                review["item_id"] = item_id
                review["timestamp"] = r["unixReviewTime"]
                try:
                    text = r["reviewText"]
                    try:
                        review["text"] = re_html.sub("", text)
                    except:
                        review["text"] = text
                except:
                    review["text"] = ""
                review["len_text"] = len(review["text"])
                reviews.append(review)

    return pd.DataFrame(reviews)
