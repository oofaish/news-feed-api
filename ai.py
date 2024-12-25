import logging

from logging import getLogger

import pandas as pd

from embedding_utils import (
    get_embeddings,
    save_embeddings_to_db,
)
from model_utils import get_recent_articles, save_tags_and_scores_to_db
from tagging_utils import get_tags_and_ai_scores
from utils import SetupClient, chunk_list, get_authenticated_client


logger = getLogger(__name__)


def add_embeddings_to_articles(df: pd.DataFrame | None, dry_run: bool = False) -> pd.DataFrame:
    if df is not None and len(df):
        logger.info(f"Found {len(df)} recent articles with no embedding")
        articles_with_embeddings = get_embeddings(df, dry_run=dry_run)
        return articles_with_embeddings
    else:
        logger.info("No recent articles with no embeddings")
        return pd.DataFrame()


def add_tags_and_ai_scores_to_articles(df: pd.DataFrame | None, dry_run: bool = False) -> pd.DataFrame:
    if df is not None and len(df):
        logger.info(f"Found {len(df)} recent articles with no ai score")
        articles_with_ai_scores = get_tags_and_ai_scores(df, dry_run=dry_run)
        return articles_with_ai_scores
    else:
        logger.info("No recent articles with no ai scores")
        return pd.DataFrame()


def update_rows(df: pd.DataFrame):
    """
    supabase does not allow me to update just some columns on some rows
    in postgres it seems I have to create a temp table to do this. Not ideal
    but do-able I guess.
    """
    entries = df[["id", "agent", "score", "ai_score"]].to_dict("records")

    client = get_authenticated_client()
    so_far = 0
    for chunk in chunk_list(entries, 50):
        so_far += len(chunk)
        logger.info(f"Saving {so_far}/{len(entries)} articles AI scores")
        client.rpc("update_articles_with_ai_results", {"data": chunk}).execute()


def main():
    with SetupClient():
        if True:
            recent_articles = get_recent_articles(["embedding2"])
            recent_articles_with_embeddings = add_embeddings_to_articles(recent_articles)
            save_embeddings_to_db(recent_articles_with_embeddings)

        recent_articles_without_ai_score = get_recent_articles(["ai_score2"])
        recent_articles_with_ai_score = add_tags_and_ai_scores_to_articles(recent_articles_without_ai_score)
        save_tags_and_scores_to_db(recent_articles_with_ai_score)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    main()
