import re

from datetime import datetime, timedelta, timezone
from random import shuffle
from typing import Optional

import numpy as np
import pandas as pd

from bs4 import BeautifulSoup
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from config import EMBEDDING_MODEL, MAX_RECENT_ARTICLES, MAX_SIZE, MAX_TITLE_SIZE, RECENT_HOURS
from openai_embedding_utils import get_embedding
from utils import ARTICLE_TABLE, SetupClient, bad_stuff, get_authenticated_client


try:
    import matplotlib.pyplot as plt
except ImportError:
    pass
from logging import getLogger


logger = getLogger(__name__)


def remove_whitespace(text):
    return " ".join(text.split())


def to_lowercase(text):
    return text.lower()


def remove_special_characters(text, remove_digits=False):
    pattern = r"[^a-zA-Z0-9\s]" if not remove_digits else r"[^a-zA-Z\s]"
    text = re.sub(pattern, "", text)
    return text


def remove_html(text: str):
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


def get_combined(article) -> Optional[str]:
    for thing in bad_stuff:
        if thing in str(article.title_short) or thing in str(article.summary_short):
            return None
    if len(article.summary_short) > 15:
        if article.title_short[:100] == article.summary_short[:100]:
            return f"A news article with title: {article.title_short}"
        return f"A news article with title: {article.title_short} and summary: {article.summary_short}"

    return f"A news article with title: {article.title_short}"


def get_articles_without_embedding_but_with_classifications() -> Optional[pd.DataFrame]:
    """
    main entry point to get all the articles that have been classified by user.

    """
    client = get_authenticated_client()
    user_marked_articles = client.table(ARTICLE_TABLE).select("*").eq("agent", "USER").is_("embedding2", "null").execute()

    df = pd.DataFrame(user_marked_articles.data)

    return prepare_articles_for_embedding(df)


def prepare_articles_for_embedding(df: pd.DataFrame) -> Optional[pd.DataFrame]:
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


def get_embeddings(df: pd.DataFrame, dry_run: bool = True, count: Optional[int] = None) -> pd.DataFrame:
    """
    hit the OpenAI API to get embeddings for the articles.
    """
    embedding_count = sum(df.embedding.notna())
    if embedding_count > 0:
        raise ValueError(f"already have {embedding_count} embeddings. Set them to null to recompute.")

    if count is not None:
        logger.info(f"limiting to {count} instead of {len(df)} articles")
        df = df.head(count).reset_index()

    if not dry_run:
        df["embedding2"] = df.combined.apply(lambda x: get_embedding(x, model=EMBEDDING_MODEL))
    else:
        logger.info(f"would have gotten embeddings for {len(df)} articles.")

    return df


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


def get_recent_articles_with_no_embedding() -> pd.DataFrame:
    client = get_authenticated_client()
    recent = datetime.now(timezone.utc) - timedelta(hours=RECENT_HOURS)
    recent_no_embedding_articles = client.table(ARTICLE_TABLE).select("*").is_("embedding2", "null").gte("created_at", recent).order("created_at", desc=True).limit(MAX_RECENT_ARTICLES).execute()

    df = pd.DataFrame(recent_no_embedding_articles.data)

    return prepare_articles_for_embedding(df)


def get_recent_articles_with_embeddings_but_no_ai_score() -> pd.DataFrame:
    client = get_authenticated_client()
    recent = datetime.now(timezone.utc) - timedelta(hours=RECENT_HOURS)
    to_score = client.table(ARTICLE_TABLE).select("*").not_.is_("embedding2", "null").is_("ai_score", "null").gte("created_at", recent).order("created_at", desc=True).execute()

    df = pd.DataFrame(to_score.data)
    return df


def score_article(row: pd.Series, model: KNeighborsClassifier) -> float:
    """
    TODO vectorize this
    """
    embedding = np.array(row.embedding)
    prediction = model.predict([embedding])[0]
    confidence_scores = model.predict_proba([embedding])[0]
    # 20 is the score I give for user classified articles
    # so if it's 1, we get 20, if it's 0.6, you get 12...
    score = max(confidence_scores) * prediction
    return score


def score_articles(df: pd.DataFrame, model: KNeighborsClassifier) -> pd.DataFrame:
    df["ai_score"] = df.apply(lambda x: score_article(x, model), axis=1)
    df.loc[df["agent"] != "USER", "agent"] = "AI"
    df.loc[df["agent"] != "USER", "score"] = df.loc[df["agent"] != "USER"]["ai_score"]

    return df


def get_articles_for_training() -> pd.DataFrame:
    """
    main entry point to get all the articles that have been classified by user and have embeddings
    """
    client = get_authenticated_client()
    user_marked_articles = client.table(ARTICLE_TABLE).select("*").eq("agent", "USER").not_.is_("embedding2", "null").execute()

    df = pd.DataFrame(user_marked_articles.data)
    return df


def train_model(df: pd.DataFrame, split: bool = False) -> KNeighborsClassifier:
    """
    train the model on the data.
    """
    knn = KNeighborsClassifier(n_neighbors=20)
    matrix = df.embedding.to_list()
    scores = df.score.to_list()

    if split:
        rs = list(range(len(matrix)))
        shuffle(rs)
        mixed_matrix = [matrix[i] for i in rs]
        mixed_scores = [scores[i] for i in rs]
        X_train, X_test, y_train, y_test = train_test_split(mixed_matrix, mixed_scores, test_size=0.01)
    else:
        X_test = X_train = matrix
        _ = y_train = scores

    knn.fit(X_train, y_train)

    if split:
        accuracy = knn.score(X_test, y_test)
        logger.info(f"Validation Accuracy: {accuracy:.2f}")

        plt.figure()
        display = PrecisionRecallDisplay.from_estimator(knn, X_test, y_test, name="knn", plot_chance_level=True)

        display.ax_.set_title("2-class Precision-Recall curve")

        plt.show()

        plt.figure()

        display = RocCurveDisplay.from_estimator(knn, X_test, y_test, name="knn", plot_chance_level=True)

        display.ax_.set_title("2-class ROC curve")

        plt.show()

    return knn


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    """
    this will hit openAI API - be aware
    """
    get_embeddings_for_classified_models = False
    train_and_save_model = True
    generate_embeddings = False
    with SetupClient():
        if get_embeddings_for_classified_models:
            done = False
            while not done:
                articles = get_articles_without_embedding_but_with_classifications()
                if articles is not None:
                    articles_with_embeddings = get_embeddings(articles, dry_run=False, count=100)
                    save_embeddings_to_db(articles_with_embeddings)
                else:
                    done = True

        if generate_embeddings:
            df = get_recent_articles_with_no_embedding()
            if df is not None and len(df):
                articles_with_embeddings = get_embeddings(
                    df,
                    dry_run=False,
                )
                save_embeddings_to_db(articles_with_embeddings)
