import datetime
import os

from logging import getLogger
from typing import Any, Optional

import feedparser

from joblib import Memory, expires_after

from config import MAX_AGE_FOR_ARTICLE_FOR_PARSSING
from embedding_utils import remove_html
from entry_processor import process
from utils import (
    ARTICLE_TABLE,
    FEED_TABLE,
    SetupClient,
    bad_stuff,
    chunk_list,
    get_authenticated_client,
)


logger = getLogger(__name__)

memory = Memory("cache_directory", verbose=0)
cache_expiry_seconds = int(os.environ.get("JOBLIB_CACHE_EXPIRY")) if os.environ.get("JOBLIB_CACHE_EXPIRY") else 0


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
        if entry["published_parsed"] > (datetime.datetime.now() - datetime.timedelta(days=MAX_AGE_FOR_ARTICLE_FOR_PARSSING)).timetuple():
            # try:
            yield process(title, entry)
            # except Exception as e:
            #     logger.info(f"Error processing {entry['link']} {e}")


def get_all_feed_urls():
    client = get_authenticated_client()
    active_feeds = client.table(FEED_TABLE).select("title,url").eq("enabled", True).execute()
    feeds = [(row["title"], row["url"]) for row in active_feeds.data]
    return feeds


def process_one_feed(url: str, title: str):
    feed = get_feed(url)
    entries = process_feed(title, feed)
    all_entries = []
    seen_links = set()

    for entry in entries:
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


def process_all_feeds():
    all_entries = []
    feed_urls = get_all_feed_urls()
    for title, feed_url in feed_urls:
        entries = process_one_feed(feed_url, title)
        all_entries.extend(entries)

    return all_entries


def save_new_entries(entries: list[dict[str, Any]]):
    client = get_authenticated_client()
    so_far = 0
    for chunk in chunk_list(entries, 50):
        so_far += len(chunk)
        logger.info(f"{so_far}/{len(entries)}")
        client.table(ARTICLE_TABLE).upsert(chunk, on_conflict="link (DO NOTHING)").execute()


def main():
    all_entries = process_all_feeds()
    save_new_entries(all_entries)


if __name__ == "__main__":
    with SetupClient():
        # main()
        # result = process_one_feed("https://hnrss.org/show?points=100&comments=25", "Hacker News")
        result = process_one_feed("https://feeds.arstechnica.com/arstechnica/index", "Ars Technica")
        pass
