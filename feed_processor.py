import datetime
import os

from logging import getLogger
from typing import Any, Optional

import feedparser

from joblib import Memory, expires_after

from embedding_utils import remove_html
from entry_processor import process
from utils import (
    ARTICLE_TABLE,
    FEED_TABLE,
    TAG_TABLE,
    SetupClient,
    bad_stuff,
    chunk_list,
    get_authenticated_client,
)


logger = getLogger(__name__)

memory = Memory("cache_directory", verbose=0)
cache_expiry_seconds = (
    int(os.environ.get("JOBLIB_CACHE_EXPIRY"))
    if os.environ.get("JOBLIB_CACHE_EXPIRY")
    else 0
)


def conditional_decorator(flag, decorator, *args, **kwargs):
    def wrapper(func):
        if flag != 0:
            return decorator(*args, **kwargs)(func)
        return func

    return wrapper


@conditional_decorator(
    cache_expiry_seconds,
    memory.cache,
    cache_validation_callback=expires_after(seconds=cache_expiry_seconds),
)
def get_feed(feed_url):
    logger.info(f"Hitting API: {feed_url}")
    feed = feedparser.parse(feed_url)
    return feed


def process_feed(title: Optional[str], feed):
    for entry in feed.entries:
        # dont bother with articles more than 5 days old
        if (
            entry["published_parsed"]
            > (datetime.datetime.now() - datetime.timedelta(days=5)).timetuple()
        ):
            try:
                yield process(title, entry)
            except Exception:
                logger.info(f"Error processing {entry['link']}")


def get_all_feed_urls():
    client = get_authenticated_client()
    active_feeds = (
        client.table(FEED_TABLE).select("title,url").eq("enabled", True).execute()
    )
    feeds = [(row["title"], row["url"]) for row in active_feeds.data]
    return feeds


def process_all_feeds():
    all_entries = []
    seen_links = set()
    feed_urls = get_all_feed_urls()
    for title, feed_url in feed_urls:
        feed = get_feed(feed_url)
        for entry in process_feed(title, feed):
            is_bad = False
            for thing in bad_stuff:
                # just filter out from title for now as I am dropping
                # stuff I shouldn't
                if thing in str(entry["title"]):
                    is_bad = True
            if not is_bad and entry["link"] not in seen_links:
                seen_links.add(entry["link"])
                try:
                    entry["summary"] = remove_html(entry["summary"])
                except Exception:
                    logger.exception("Failed to remove html from summary")
                all_entries.append(entry)

    return all_entries


def save_new_entries(entries: list[dict[str, Any]]):
    client = get_authenticated_client()
    so_far = 0
    for chunk in chunk_list(entries, 50):
        so_far += len(chunk)
        logger.info(f"{so_far}/{len(entries)}")
        client.table(ARTICLE_TABLE).upsert(
            chunk, on_conflict="link (DO NOTHING)"
        ).execute()


def save_new_tags(tags: list[str]):
    client = get_authenticated_client()
    so_far = 0
    for chunk in chunk_list(tags, 50):
        so_far += len(chunk)
        logger.info(f"{so_far}/{len(tags)}")
        client.table(TAG_TABLE).upsert(
            [{"name": x} for x in chunk], on_conflict="name (DO NOTHING)"
        ).execute()


def main():
    all_entries = process_all_feeds()
    all_tags = list(
        set(
            sum(
                [
                    x.get("tags")
                    for x in all_entries
                    if "tags" in x and x["tags"] is not None
                ],
                [],
            )
        )
    )
    all_tags = sorted(all_tags)
    save_new_tags(all_tags)
    save_new_entries(all_entries)


if __name__ == "__main__":
    with SetupClient():
        main()
