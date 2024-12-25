import datetime
import os

from logging import getLogger
from typing import Any, Iterator

import feedparser
import yaml

from joblib import Memory, expires_after

from config import MAX_AGE_FOR_ARTICLE_FOR_PARSSING, Publication
from entry_processor import process
from model_utils import remove_html
from utils import (
    ARTICLE_TABLE,
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


def process_feed(publication: Publication, feed: feedparser.FeedParserDict) -> Iterator[dict[str, Any]]:
    for entry in feed.entries:
        # dont bother with articles more than 5 days old
        if entry["published_parsed"] > (datetime.datetime.now() - datetime.timedelta(days=MAX_AGE_FOR_ARTICLE_FOR_PARSSING)).timetuple():
            # try:
            yield process(publication, entry)
            # except Exception as e:
            #     logger.info(f"Error processing {entry['link']} {e}")


def get_all_feed_urls() -> list[tuple[Publication, list[str]]]:
    # load the yaml file of publications.yaml
    with open("publications.yaml", "r") as f:
        publications = yaml.safe_load(f)

    result = [(Publication(p["publication"]), p["urls"]) for p in publications if p["enabled"]]

    return result
    # client = get_authenticated_client()
    # active_feeds = client.table(FEED_TABLE).select("title,url").eq("enabled", True).execute()
    # feeds = [(row["title"], row["url"]) for row in active_feeds.data]
    # return feeds


def process_one_feed(url: str, publication: Publication, seen_links: set[str]):
    feed = get_feed(url)
    entries = process_feed(publication, feed)
    all_entries = []

    for entry in entries:
        entry["publication"] = publication.value
        # convert tags to list
        entry["tags"] = list(entry["tags"])
        is_bad = False
        for thing in bad_stuff:
            # # just filter out from title for now as I am dropping
            # # stuff I shouldn't
            # if thing in str(entry["title"]):
            #     is_bad = True
            is_bad = False
        if not is_bad and entry["link"] not in seen_links:
            seen_links.add(entry["link"])
            try:
                entry["summary"] = remove_html(entry["summary"])
            except Exception as e:
                logger.exception(f"Failed to remove html from summary: {e}")
            all_entries.append(entry)

    return all_entries


def process_all_feeds():
    all_entries = []
    all_feed_urls = get_all_feed_urls()
    publication: Publication
    seen_entry_urls = set()
    for publication, feed_urls in all_feed_urls:
        for url in feed_urls:
            entries = process_one_feed(url, publication, seen_entry_urls)
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
    # feeds = get_all_feed_urls()
    main()
    pass
    # with SetupClient():
    #     # main()
    #     # result = process_one_feed("https://hnrss.org/show?points=100&comments=25", "Hacker News")
    #     result = process_one_feed("https://feeds.arstechnica.com/arstechnica/index", "Ars Technica")
    #     pass
