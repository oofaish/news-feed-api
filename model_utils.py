import re

from datetime import datetime, timedelta, timezone
from logging import getLogger

import pandas as pd

from bs4 import BeautifulSoup

from config import MAX_RECENT_ARTICLES, MAX_SIZE, MAX_TITLE_SIZE, RECENT_HOURS
from utils import ARTICLE_TABLE, bad_stuff, get_authenticated_client


logger = getLogger(__name__)


def remove_whitespace(text):
    return " ".join(text.split())


def to_lowercase(text):
    return text.lower()


def remove_special_characters(text, remove_digits=False):
    pattern = r"[^a-zA-Z0-9\s]" if not remove_digits else r"[^a-zA-Z\s]"
    text = re.sub(pattern, "", text)
    return text


def remove_html(text: str | None) -> str:
    if text is None:
        return ""
    return BeautifulSoup(text, "html.parser").get_text()


def cut_short(text: str, size) -> str:
    if len(text) < size:
        return text

    last_space = next(i for i, x in enumerate(reversed(text[:size])) if x == " ")

    return text[: size - last_space].strip() + "..."


def get_short_article_title(article) -> str:
    try:
        original_title = article.title.strip() if article.title else ""
        title = remove_whitespace(remove_html(original_title)).split("|")[0].strip()
        title = cut_short(title, MAX_TITLE_SIZE)

        if len(title) < len(original_title):
            logger.debug(f"Title shortened from {len(original_title)} to {len(title)} chars")

        return title
    except Exception as e:
        logger.exception(f"Error processing article title: {e}")
        return "Error processing title"


def get_short_summary(article) -> str:
    try:
        if article.summary is None:
            logger.debug("Article has no summary")
            return ""

        original_summary = article.summary.strip()
        summary = remove_whitespace(remove_html(original_summary)).split("|")[0]
        summary = cut_short(summary, MAX_SIZE)

        if len(summary) < len(original_summary):
            logger.debug(f"Summary shortened from {len(original_summary)} to {len(summary)} chars")

        return summary
    except Exception as e:
        logger.exception(f"Error processing article summary: {e}")
        return "Error processing summary"


def get_combined(article, use_short: bool = False) -> str | None:
    article_id = getattr(article, "id", "unknown")
    logger.debug(f"Combining text for article ID: {article_id}")

    try:
        # Check for bad content
        for thing in bad_stuff:
            if thing in str(article.title_short) or thing in str(article.summary_short):
                logger.info(f"Article with potentially problematic content in title/summary: {article.get('title_short', 'unknown')}")

        # Select appropriate fields based on use_short flag
        if use_short:
            title = article.title_short
            summary = article.summary_short
            logger.debug("Using shortened title and summary")
        else:
            title = article.title
            summary = article.summary
            logger.debug("Using full title and summary")

        publication = article.publication
        logger.debug(f"Combining text for article from {publication}")

        # Build the combined text
        if summary == title:
            logger.debug("Summary matches title, using simplified format")
            full_text = f"""
                News article from {publication}
                Title: {title}
            """
        else:
            full_text = f"""
                News article from {publication}
                Title: {title}
                Summary: {summary}
            """

        # Add author if available
        if article.author is not None:
            logger.debug(f"Adding author: {article.author}")
            full_text += f"""
                Author: {article.author}
            """

        combined_text = full_text.strip()
        logger.debug(f"Combined text created, length: {len(combined_text)} chars")
        return combined_text
    except Exception as e:
        logger.exception(f"Error combining text for article {article_id}: {e}")
        return None


def prepare_articles_for_models(df: pd.DataFrame) -> pd.DataFrame | None:
    if len(df) == 0:
        logger.info("No articles found to prepare for models")
        return None

    logger.info(f"Preparing {len(df)} articles for models")

    try:
        # Process titles
        logger.debug("Processing article titles")
        start_time = pd.Timestamp.now()
        df["title_short"] = df.apply(get_short_article_title, axis=1)
        logger.debug(f"Processed titles in {(pd.Timestamp.now() - start_time).total_seconds():.2f} seconds")

        # Process summaries
        logger.debug("Processing article summaries")
        start_time = pd.Timestamp.now()
        df["summary_short"] = df.apply(get_short_summary, axis=1)
        logger.debug(f"Processed summaries in {(pd.Timestamp.now() - start_time).total_seconds():.2f} seconds")

        # Combine text
        logger.debug("Combining article text")
        start_time = pd.Timestamp.now()
        df["combined"] = df.apply(get_combined, axis=1)
        logger.debug(f"Combined text in {(pd.Timestamp.now() - start_time).total_seconds():.2f} seconds")

        # Filter out articles with no combined text
        no_na_df = df[df["combined"].notna()]
        filtered_count = len(df) - len(no_na_df)

        if filtered_count > 0:
            logger.info(f"Filtered out {filtered_count} articles with no combined text")

        if len(no_na_df) == 0:
            logger.warning("No valid articles found after filtering")
            return None

        logger.info(f"Successfully prepared {len(no_na_df)} articles for models")
        return no_na_df
    except Exception as e:
        logger.exception(f"Error preparing articles for models: {e}")
        return None


def get_recent_articles(
    null_columns: list[str] | None = None,
    recent_hours: int = RECENT_HOURS,
    max_recent_articles: int = MAX_RECENT_ARTICLES,
) -> pd.DataFrame:
    logger.info(f"Retrieving recent articles from the last {recent_hours} hours, limit: {max_recent_articles}")

    try:
        client = get_authenticated_client()
        recent = datetime.now(timezone.utc) - timedelta(hours=recent_hours)

        # Build query
        recent_articles_query = client.table(ARTICLE_TABLE).select("*")

        # Add null column filters if specified
        if null_columns is not None:
            logger.info(f"Filtering for articles with NULL values in columns: {', '.join(null_columns)}")
            for column in null_columns:
                recent_articles_query = recent_articles_query.is_(column, "null")

        # Execute query with time filter, ordering, and limit
        start_time = datetime.now(timezone.utc)
        query_result = recent_articles_query.gte("created_at", recent).order("created_at", desc=True).limit(max_recent_articles).execute()
        query_time = (datetime.now(timezone.utc) - start_time).total_seconds()

        # Process results
        article_count = len(query_result.data)
        logger.info(f"Retrieved {article_count} articles in {query_time:.2f} seconds")

        if article_count == 0:
            logger.info("No articles found matching the criteria")
            return pd.DataFrame()

        # Convert to DataFrame and prepare for models
        df = pd.DataFrame(query_result.data)
        return prepare_articles_for_models(df)
    except Exception as e:
        logger.exception(f"Error retrieving recent articles: {e}")
        return pd.DataFrame()


def save_embeddings_to_db(df: pd.DataFrame) -> None:
    """
    save the embeddings to the database.
    """
    if df.empty:
        logger.info("No embeddings to save to database")
        return

    article_count = len(df)
    logger.info(f"Saving {article_count} embeddings to database")

    try:
        client = get_authenticated_client()
        start_time = pd.Timestamp.now()
        success_count = 0
        error_count = 0

        for i, (_, row) in enumerate(df.iterrows(), 1):
            try:
                client.table(ARTICLE_TABLE).update({"embedding2": row.embedding2}).eq("id", row.id).execute()
                success_count += 1
                if i % 10 == 0 or i == article_count:
                    logger.info(f"Progress: {i}/{article_count} embeddings saved ({(i/article_count)*100:.1f}%)")
            except Exception as e:
                error_count += 1
                logger.error(f"Error saving embedding for article {row.id}: {e}")

        duration = (pd.Timestamp.now() - start_time).total_seconds()
        logger.info(f"Completed saving {success_count}/{article_count} embeddings in {duration:.2f} seconds, {error_count} errors")
    except Exception as e:
        logger.exception(f"Error in save_embeddings_to_db: {e}")


def save_tags_and_scores_to_db(df: pd.DataFrame) -> None:
    """
    save the tags and scores to the database.
    """
    if df.empty:
        logger.info("No tags and scores to save to database")
        return

    article_count = len(df)
    logger.info(f"Saving tags and scores for {article_count} articles to database")

    try:
        client = get_authenticated_client()
        start_time = pd.Timestamp.now()
        success_count = 0
        error_count = 0

        for i, (_, row) in enumerate(df.iterrows(), 1):
            try:
                update_data = {"ai_score2": row.ai_score2, "tags_topic": row.tags_topic, "tags_mood": row.tags_mood, "tags_scope": row.tags_scope, "score": row.score, "agent": row.agent}

                client.table(ARTICLE_TABLE).update(update_data).eq("id", row.id).execute()
                success_count += 1

                if i % 10 == 0 or i == article_count:
                    logger.info(f"Progress: {i}/{article_count} articles updated with tags and scores ({(i/article_count)*100:.1f}%)")
            except Exception as e:
                error_count += 1
                logger.error(f"Error saving tags and scores for article {row.id}: {e}")

        duration = (pd.Timestamp.now() - start_time).total_seconds()
        logger.info(f"Completed saving tags and scores for {success_count}/{article_count} articles in {duration:.2f} seconds, {error_count} errors")
    except Exception as e:
        logger.exception(f"Error in save_tags_and_scores_to_db: {e}")
