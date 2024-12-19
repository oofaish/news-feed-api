from logging import getLogger
from typing import Optional

import pandas as pd

from config import EMBEDDING_MODEL
from model_utils import get_recent_articles, prepare_articles_for_models
from openai_embedding_utils import get_embedding
from utils import ARTICLE_TABLE, SetupClient, get_authenticated_client


logger = getLogger(__name__)


def get_articles_without_embedding_but_with_classifications() -> Optional[pd.DataFrame]:
    """
    main entry point to get all the articles that have been classified by user.

    """
    client = get_authenticated_client()
    user_marked_articles = client.table(ARTICLE_TABLE).select("*").eq("agent", "USER").is_("embedding2", "null").execute()

    df = pd.DataFrame(user_marked_articles.data)

    return prepare_articles_for_models(df)


def get_embeddings(df: pd.DataFrame, dry_run: bool = True, count: Optional[int] = None) -> pd.DataFrame:
    """
    hit the OpenAI API to get embeddings for the articles.
    """
    embedding_count = sum(df.embedding2.notna())
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


def get_articles_for_training() -> pd.DataFrame:
    """
    main entry point to get all the articles that have been classified by user and have embeddings
    """
    client = get_authenticated_client()
    user_marked_articles = client.table(ARTICLE_TABLE).select("*").eq("agent", "USER").not_.is_("embedding2", "null").execute()

    df = pd.DataFrame(user_marked_articles.data)
    return df


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    """
    this will hit openAI API - be aware
    """
    get_embeddings_for_classified_models = False
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
            df = get_recent_articles(null_columns=["embedding2"])
            if df is not None and len(df):
                articles_with_embeddings = get_embeddings(
                    df,
                    dry_run=False,
                )
                save_embeddings_to_db(articles_with_embeddings)
