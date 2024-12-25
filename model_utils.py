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
    title = remove_whitespace(remove_html(article.title.strip())).split("|")[0].strip()

    title = cut_short(title, MAX_TITLE_SIZE)
    return title


def get_short_summary(article) -> str:
    summary = remove_whitespace(remove_html(article.summary.strip())).split("|")[0] if article.summary is not None else ""
    summary = cut_short(summary, MAX_SIZE)
    return summary


def get_combined(article, use_short: bool = False) -> str | None:
    for thing in bad_stuff:
        if thing in str(article.title_short) or thing in str(article.summary_short):
            logger.info(f"Used to skip article with bad stuff in title or summary {article['title_short']}")

    if use_short:
        title = article.title_short
        summary = article.summary_short
    else:
        title = article.title
        summary = article.summary

    publication = article.publication

    if summary == title:
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
    # put the author if it's not null
    if article.author is not None:
        full_text += f"""
            Author: {article.author}
        """

    return full_text.strip()


def prepare_articles_for_models(df: pd.DataFrame) -> pd.DataFrame | None:
    if len(df) == 0:
        logger.info("no articles found")
        return None

    df["title_short"] = df.apply(get_short_article_title, axis=1)
    df["summary_short"] = df.apply(get_short_summary, axis=1)
    df["combined"] = df.apply(get_combined, axis=1)

    no_na_df = df[df["combined"].notna()]

    if len(no_na_df) == 0:
        logger.info("no good articles found")
        return None

    return no_na_df


def get_recent_articles(
    null_columns: list[str] | None = None,
    recent_hours: int = RECENT_HOURS,
    max_recent_articles: int = MAX_RECENT_ARTICLES,
) -> pd.DataFrame:
    client = get_authenticated_client()
    recent = datetime.now(timezone.utc) - timedelta(hours=recent_hours)
    recent_no_ai_score_articles = client.table(ARTICLE_TABLE).select("*")
    if null_columns is not None:
        for column in null_columns:
            recent_no_ai_score_articles = recent_no_ai_score_articles.is_(column, "null")

    recent_no_ai_something_articles = recent_no_ai_score_articles.gte("created_at", recent).order("created_at", desc=True).limit(max_recent_articles).execute()

    df = pd.DataFrame(recent_no_ai_something_articles.data)

    return prepare_articles_for_models(df)


def save_embeddings_to_db(df: pd.DataFrame) -> None:
    """
    save the embeddings to the database.
    """
    client = get_authenticated_client()
    logger.info("updating %d embeddings", len(df))
    for _, row in df.iterrows():
        client.table(ARTICLE_TABLE).update({"embedding2": row.embedding2}).eq("id", row.id).execute()
        logger.info("updated embedding for %s", row.id)

    logger.info("done updating embeddings")


def save_tags_and_scores_to_db(df: pd.DataFrame) -> None:
    """
    save the tags and scores to the database.
    """
    client = get_authenticated_client()
    logger.info("updating %d tags and scores", len(df))
    for _, row in df.iterrows():
        client.table(ARTICLE_TABLE).update({"ai_score2": row.ai_score2, "tags_topic": row.tags_topic, "tags_mood": row.tags_mood, "tags_scope": row.tags_scope}).eq("id", row.id).execute()
        logger.info("updated tags and scores for %s", row.id)

    logger.info("done updating tags and scores")
