from logging import getLogger

import pandas as pd

from tag_builder import analyze_content


logger = getLogger(__name__)

columns_being_set = {
    "ai_score2",
    "tags_topic",
    "tags_mood",
    "tags_scope",
}


def process_row(row: pd.Series) -> pd.Series:
    """
    Process a single row of the DataFrame to get AI analysis results.
    """
    result = analyze_content(row["combined"])
    if result.error is None:
        row["ai_score2"] = result.score
        if row["agent"] != "USER":
            row["score"] = result.score
            row["agent"] = "AI"
        row["tags_topic"] = result.topic
        row["tags_mood"] = result.mood
        row["tags_scope"] = result.scope
        row["scoring_status"] = "SUCCESS"
    else:
        row["scoring_status"] = f"FAILED: {result.error}"
    return row


def get_tags_and_ai_scores(df: pd.DataFrame, dry_run: bool = True, count: int | None = None) -> pd.DataFrame:
    """
    hit the OpenAI API to get embeddings for the articles.
    """
    for column in columns_being_set:
        not_null_count = sum(df[column].notna())
        if not_null_count > 0:
            raise ValueError(f"already have {not_null_count} {column}. Set all columns {columns_being_set} to null to recompute.")

    if count is not None:
        logger.info(f"limiting to {count} instead of {len(df)} articles")
        df = df.head(count).reset_index()

    if not dry_run:
        df = df.apply(process_row, axis=1)
    else:
        logger.info(f"would have gotten tags and score for {len(df)} articles.")

    return df
