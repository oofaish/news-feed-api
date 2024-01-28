from logging import getLogger

import pandas as pd

from embedding_utils import (
    get_embeddings,
    get_model_from_database,
    get_recent_articles_with_embeddings_but_no_ai_score,
    get_recent_articles_with_no_embedding,
    save_embeddings_to_db,
    score_articles,
)
from utils import SetupClient, chunk_list, get_authenticated_client


logger = getLogger(__name__)


def add_embeddings_to_recent_articles():
    df = get_recent_articles_with_no_embedding()
    if df is not None and len(df):
        logger.info(f"Found {len(df)} recent articles with no embeddings")
        articles_with_embeddings = get_embeddings(df, dry_run=False)
        save_embeddings_to_db(articles_with_embeddings)
    else:
        logger.info("No recent articles with no embeddings")


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
        add_embeddings_to_recent_articles()
        df = get_recent_articles_with_embeddings_but_no_ai_score()
        if len(df):
            model = get_model_from_database()
            scored_articles = score_articles(df, model)
            logger.info(f"Scored {len(scored_articles)} articles")
            update_rows(scored_articles)

        else:
            logger.info("No articles to score")


if __name__ == "__main__":
    main()
