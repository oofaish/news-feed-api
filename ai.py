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
        article_count = len(df)
        logger.info(f"Found {article_count} recent articles with no embedding")
        try:
            start_time = pd.Timestamp.now()
            articles_with_embeddings = get_embeddings(df, dry_run=dry_run)
            duration = (pd.Timestamp.now() - start_time).total_seconds()

            success_count = sum(articles_with_embeddings["embedding2"].notna()) if not dry_run else 0
            logger.info(f"Successfully added embeddings to {success_count}/{article_count} articles in {duration:.2f} seconds")
            return articles_with_embeddings
        except Exception as e:
            logger.exception(f"Error adding embeddings to articles: {e}")
            return pd.DataFrame()
    else:
        logger.info("No recent articles with no embeddings")
        return pd.DataFrame()


def add_tags_and_ai_scores_to_articles(df: pd.DataFrame | None, dry_run: bool = False) -> pd.DataFrame:
    if df is not None and len(df):
        article_count = len(df)
        logger.info(f"Found {article_count} recent articles with no ai score")
        try:
            start_time = pd.Timestamp.now()
            articles_with_ai_scores = get_tags_and_ai_scores(df, dry_run=dry_run)
            duration = (pd.Timestamp.now() - start_time).total_seconds()

            if not dry_run:
                success_count = sum(articles_with_ai_scores["scoring_status"] == "SUCCESS") if "scoring_status" in articles_with_ai_scores.columns else 0
                error_count = sum(articles_with_ai_scores["scoring_status"].str.startswith("FAILED")) if "scoring_status" in articles_with_ai_scores.columns else 0
                logger.info(f"Added AI scores to {success_count}/{article_count} articles in {duration:.2f} seconds, {error_count} errors")
            else:
                logger.info(f"Would have added AI scores to {article_count} articles (dry run)")

            return articles_with_ai_scores
        except Exception as e:
            logger.exception(f"Error adding tags and AI scores to articles: {e}")
            return pd.DataFrame()
    else:
        logger.info("No recent articles with no ai scores")
        return pd.DataFrame()


def update_rows(df: pd.DataFrame):
    """
    supabase does not allow me to update just some columns on some rows
    in postgres it seems I have to create a temp table to do this. Not ideal
    but do-able I guess.
    """
    if df.empty:
        logger.info("No rows to update")
        return

    entries = df[["id", "agent", "score", "ai_score"]].to_dict("records")
    total_entries = len(entries)
    logger.info(f"Preparing to update {total_entries} articles with AI scores")

    client = get_authenticated_client()
    so_far = 0
    success_count = 0
    error_count = 0
    chunk_count = 0
    total_chunks = len(list(chunk_list(entries, 50)))
    start_time = pd.Timestamp.now()

    try:
        for chunk in chunk_list(entries, 50):
            chunk_count += 1
            chunk_size = len(chunk)
            so_far += chunk_size
            logger.info(f"Updating chunk {chunk_count}/{total_chunks}: {so_far}/{total_entries} articles")

            try:
                client.rpc("update_articles_with_ai_results", {"data": chunk}).execute()
                success_count += chunk_size
                logger.info(f"Successfully updated chunk {chunk_count} with {chunk_size} articles")
            except Exception as e:
                error_count += chunk_size
                logger.error(f"Failed to update chunk {chunk_count}: {e}")

        duration = (pd.Timestamp.now() - start_time).total_seconds()
        logger.info(f"Completed updating {success_count} articles with AI scores in {duration:.2f} seconds, {error_count} errors")
    except Exception as e:
        logger.error(f"Error during update operation: {e}")


def main():
    logger.info("Starting AI processing main function")
    total_start_time = pd.Timestamp.now()

    try:
        with SetupClient():
            # Process embeddings
            logger.info("Starting embedding processing")
            embedding_start_time = pd.Timestamp.now()

            try:
                recent_articles = get_recent_articles(["embedding2"])
                if recent_articles is not None and not recent_articles.empty:
                    logger.info(f"Found {len(recent_articles)} articles needing embeddings")
                    recent_articles_with_embeddings = add_embeddings_to_articles(recent_articles)
                    if not recent_articles_with_embeddings.empty:
                        save_embeddings_to_db(recent_articles_with_embeddings)
                else:
                    logger.info("No articles found needing embeddings")

                embedding_duration = (pd.Timestamp.now() - embedding_start_time).total_seconds()
                logger.info(f"Completed embedding processing in {embedding_duration:.2f} seconds")
            except Exception as e:
                logger.exception(f"Error during embedding processing: {e}")

            # Process AI scores and tags
            logger.info("Starting AI scoring and tagging")
            scoring_start_time = pd.Timestamp.now()

            try:
                recent_articles_without_ai_score = get_recent_articles(["ai_score2"])
                if recent_articles_without_ai_score is not None and not recent_articles_without_ai_score.empty:
                    logger.info(f"Found {len(recent_articles_without_ai_score)} articles needing AI scores")
                    recent_articles_with_ai_score = add_tags_and_ai_scores_to_articles(recent_articles_without_ai_score)
                    if not recent_articles_with_ai_score.empty:
                        save_tags_and_scores_to_db(recent_articles_with_ai_score)
                else:
                    logger.info("No articles found needing AI scores")

                scoring_duration = (pd.Timestamp.now() - scoring_start_time).total_seconds()
                logger.info(f"Completed AI scoring and tagging in {scoring_duration:.2f} seconds")
            except Exception as e:
                logger.exception(f"Error during AI scoring and tagging: {e}")
    except Exception as e:
        logger.exception(f"Error in AI processing main function: {e}")

    total_duration = (pd.Timestamp.now() - total_start_time).total_seconds()
    logger.info(f"Completed AI processing in {total_duration:.2f} seconds")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
