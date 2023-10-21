from enum import Enum
from typing import Union

from utils import ARTICLE_TABLE, TAG_TABLE, SetupClient, chunk_list, get_authenticated_client


# from datetime import datetime, timezone


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


def is_disliked_by_tag(article_tags, negative_tags) -> Union[bool, str]:
    if article_tags is None:
        return False

    for tag in article_tags:
        if tag in negative_tags:
            return tag

    return False


def is_disliked_title_or_summary(title, summary, negative_tags) -> Union[bool, str]:
    if summary is None:
        summary = ""

    if title is not None:
        for tag in negative_tags:
            if tag in title or tag in summary:
                return tag

    return False


def run_filters():
    # load all rows from supabase where agent is null
    client = get_authenticated_client()
    query = client.table(ARTICLE_TABLE).select("*").is_("agent", "null").execute()

    updated_rows = []
    negative_tags = get_negative_tags()
    # now = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S%z')

    for row in query.data:
        article_tags = row["tags"]

        if tag := is_disliked_by_tag(article_tags, negative_tags):
            updates = {
                "agent": AgentType.TAG.value,
                "score": -10,
                "reason": tag,
                "id": row["id"],
                # 'updated_at': now,
            }
            updated_rows.append(updates)

        elif tag := is_disliked_title_or_summary(row["title"], row["summary"], negative_tags):
            updates = {
                "agent": AgentType.TITLE.value,
                "score": -5,
                "reason": tag,
                "id": row["id"],
                # 'updated_at': now,
            }
            updated_rows.append(updates)
            # row['agent'] = AgentType.TITLE.value
            # row['score'] = -5
            # updated_rows.append(row)
        else:
            updates = {
                "agent": AgentType.NONE.value,
                "score": row["score"],
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
        print(f"{so_far}/{len(entries)}")
        client.rpc("update_articles_with_agent_results", {"data": chunk}).execute()


def main():
    entries = run_filters()
    update_rows(entries)


if __name__ == "__main__":
    with SetupClient():
        main()
