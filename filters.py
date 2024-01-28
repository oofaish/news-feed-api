from enum import Enum

# from datetime import datetime, timezone
from logging import getLogger
from typing import Optional

from utils import (
    ARTICLE_TABLE,
    TAG_TABLE,
    SetupClient,
    chunk_list,
    get_authenticated_client,
)


logger = getLogger(__name__)


class AgentType(Enum):
    NONE = "NONE"
    TAG = "TAG"
    TITLE = "TITLE"
    USER = "USER"
    CHATGPT = "CHATGPT"


def get_negative_tags():
    client = get_authenticated_client()
    negative_tags_query = client.table(TAG_TABLE).select("*").lt("score", 0).execute()

    negative_tags = [row["name"] for row in negative_tags_query.data]

    return negative_tags


def get_positive_tags():
    client = get_authenticated_client()
    positive_tags_query = client.table(TAG_TABLE).select("*").gt("score", 0).execute()

    positive_tags = [row["name"] for row in positive_tags_query.data]

    return positive_tags


def is_marked_by_tag(
    article_tags, negative_tags, positive_tags
) -> tuple[int, Optional[str]]:
    if article_tags is None:
        return 0, None

    negative_tag = None
    positive_tag = None
    for tag in article_tags:
        if tag in negative_tags:
            negative_tag = tag
        elif tag in positive_tags:
            positive_tag = tag

    if negative_tag is not None and positive_tag is None:
        return -10, negative_tag

    if positive_tag is not None and negative_tag is None:
        return 10, positive_tag

    return 0, None


def is_marked_title_or_summary(
    title, summary, negative_tags, positive_tags
) -> tuple[int, Optional[str]]:
    if summary is None:
        summary = ""

    negative_tag = None
    positive_tag = None
    if title is not None:
        if summary is not None:
            search_text = title + summary
        else:
            search_text = title

        search_text = search_text.title()

        for tag in negative_tags:
            if tag in search_text:
                negative_tag = tag
                break
        for tag in positive_tags:
            if tag in search_text:
                positive_tag = tag
                break

    if negative_tag is not None and positive_tag is None:
        return -5, negative_tag

    if positive_tag is not None and negative_tag is None:
        return 5, positive_tag
    return 0, None


def run_filters(run_on_all_positives=False):
    # load all rows from supabase where agent is null
    client = get_authenticated_client()
    if run_on_all_positives:
        query = (
            client.table(ARTICLE_TABLE)
            .select("*")
            .neq("agent", "USER")
            .gte("score", 0)
            .execute()
        )
    else:
        query = client.table(ARTICLE_TABLE).select("*").is_("agent", "null").execute()
    # query = client.table(ARTICLE_TABLE).select("*").eq("id", 30936).execute()

    updated_rows = []
    negative_tags = get_negative_tags()
    positive_tags = get_positive_tags()
    # now = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S%z')

    for row in query.data:
        article_tags = row["tags"]
        if row["agent"] is not None and row["agent"].lower() == "user":
            raise ValueError(f"Something is wrong - about to update {row}")

        score, tag = is_marked_by_tag(article_tags, negative_tags, positive_tags)

        if score != 0:
            updates = {
                "agent": AgentType.TAG.value,
                "score": score,
                "reason": tag,
                "id": row["id"],
            }
            updated_rows.append(updates)
            continue

        score, tag = is_marked_title_or_summary(
            row["title"], row["summary"], negative_tags, positive_tags
        )
        if score != 0:
            updates = {
                "agent": AgentType.TITLE.value,
                "score": score,
                "reason": tag,
                "id": row["id"],
                # 'updated_at': now,
            }
            updated_rows.append(updates)
            continue
            # row['agent'] = AgentType.TITLE.value
            # row['score'] = -5
            # updated_rows.append(row)

        updates = {
            "agent": AgentType.NONE.value,
            "score": score,
            "id": row["id"],
            # 'reason': '',
            # 'updated_at': now,
        }

        updated_rows.append(updates)
        # row['agent'] = AgentType.NONE.value
        # row['score'] = 0
        # updated_rows.append(row)

    return updated_rows


def update_rows(entries):
    """
    supabase does not allow me to update just some columns on some rows
    in postgres it seems I have to create a temp table to do this. Not ideal
    but do-able I guess.
    """
    client = get_authenticated_client()
    so_far = 0
    for chunk in chunk_list(entries, 50):
        so_far += len(chunk)
        logger.info(f"{so_far}/{len(entries)}")
        client.rpc("update_articles_with_agent_results", {"data": chunk}).execute()


def main(run_on_all_positives=False):
    entries = run_filters(run_on_all_positives=run_on_all_positives)
    update_rows(entries)


if __name__ == "__main__":
    with SetupClient():
        main(run_on_all_positives=False)
