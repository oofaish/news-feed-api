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
    article_id = row.get("id", "unknown")
    logger.debug(f"Processing article ID: {article_id} for tagging and scoring")

    try:
        start_time = pd.Timestamp.now()
        combined_text_length = len(row["combined"]) if pd.notna(row["combined"]) else 0
        logger.debug(f"Article {article_id} has {combined_text_length} chars of combined text")

        result = analyze_content(row["combined"])
        duration = (pd.Timestamp.now() - start_time).total_seconds()

        if result.error is None:
            # Log successful tagging details
            topic_tags = result.topic if result.topic else []
            mood_tags = result.mood if result.mood else []
            scope_tags = result.scope if result.scope else []
            score = result.score if result.score is not None else "N/A"

            logger.info(
                f"Successfully tagged article {article_id} in {duration:.2f}s - " f"Score: {score}, Topics: {', '.join(topic_tags)}, " f"Mood: {', '.join(mood_tags)}, Scope: {', '.join(scope_tags)}"
            )

            # Update row with results
            row["ai_score2"] = result.score
            if row["agent"] != "USER":
                row["score"] = result.score
                row["agent"] = "AI"
            row["tags_topic"] = result.topic
            row["tags_mood"] = result.mood
            row["tags_scope"] = result.scope
            row["scoring_status"] = "SUCCESS"
        else:
            # Log error details
            logger.error(f"Failed to tag article {article_id}: {result.error}")
            row["scoring_status"] = f"FAILED: {result.error}"
    except Exception as e:
        logger.exception(f"Unexpected error processing article {article_id}: {e}")
        row["scoring_status"] = f"FAILED: Unexpected error: {str(e)}"

    return row


def get_tags_and_ai_scores(df: pd.DataFrame, dry_run: bool = True, count: int | None = None) -> pd.DataFrame:
    """
    Get tags and AI scores for articles by calling the OpenAI API.
    """
    # Check if we already have data in the columns we're going to set
    logger.info(f"Checking {len(df)} articles for existing tag and score data")
    for column in columns_being_set:
        not_null_count = sum(df[column].notna())
        if not_null_count > 0:
            logger.warning(f"Found {not_null_count} articles with existing {column} values")
            raise ValueError(f"Already have {not_null_count} {column}. Set all columns {columns_being_set} to null to recompute.")

    # Apply count limit if specified
    if count is not None:
        logger.info(f"Limiting processing to {count} articles instead of {len(df)} total articles")
        df = df.head(count).reset_index()

    total_articles = len(df)
    logger.info(f"Preparing to process {total_articles} articles for tagging and scoring")

    if not dry_run:
        logger.info(f"Starting processing of {total_articles} articles")
        start_time = pd.Timestamp.now()

        # Process each row to get tags and scores
        df = df.apply(process_row, axis=1)

        # Calculate statistics
        duration = (pd.Timestamp.now() - start_time).total_seconds()
        success_count = sum(df["scoring_status"] == "SUCCESS")
        failure_count = total_articles - success_count
        avg_time = duration / total_articles if total_articles > 0 else 0

        logger.info(f"Completed tagging and scoring {total_articles} articles in {duration:.2f} seconds")
        logger.info(f"Success: {success_count}, Failures: {failure_count}, Avg time per article: {avg_time:.2f}s")
    else:
        logger.info(f"DRY RUN: Would have processed {total_articles} articles for tagging and scoring")

    return df
