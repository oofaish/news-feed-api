import os

from typing import Any

import feedparser

from joblib import Memory, expires_after

from entry_processor import process
from utils import ARTICLE_TABLE, FEED_TABLE, TAG_TABLE, SetupClient, chunk_list, get_authenticated_client


memory = Memory("cache_directory", verbose=0)


@memory.cache(cache_validation_callback=expires_after(seconds=int(os.environ.get("JOBLIB_CACHE_EXPIRY"))))
def get_feed(feed_url):
    print(f"Hitting API: {feed_url}")
    feed = feedparser.parse(feed_url)
    return feed


def process_feed(feed):
    for entry in feed.entries:
        yield process(entry)


def get_all_feed_urls():
    client = get_authenticated_client()
    active_feeds = client.table(FEED_TABLE).select("url").eq("enabled", True).execute()
    feeds = [row["url"] for row in active_feeds.data]
    return feeds


def process_all_feeds():
    all_entries = []
    seen_links = set()
    feed_urls = get_all_feed_urls()
    for feed_url in feed_urls:
        feed = get_feed(feed_url)
        for entry in process_feed(feed):
            if entry["link"] not in seen_links:
                seen_links.add(entry["link"])
                all_entries.append(entry)

    return all_entries


def save_new_entries(entries: list[dict[str, Any]]):
    client = get_authenticated_client()
    so_far = 0
    for chunk in chunk_list(entries, 50):
        so_far += len(chunk)
        print(f"{so_far}/{len(entries)}")
        client.table(ARTICLE_TABLE).upsert(chunk, on_conflict="link (DO NOTHING)").execute()


def save_new_tags(tags: list[str]):
    client = get_authenticated_client()
    so_far = 0
    for chunk in chunk_list(tags, 50):
        so_far += len(chunk)
        print(f"{so_far}/{len(tags)}")
        client.table(TAG_TABLE).upsert([{"name": x} for x in chunk], on_conflict="name (DO NOTHING)").execute()


def main():
    all_entries = process_all_feeds()
    all_tags = list(set(sum([x.get("tags") for x in all_entries if "tags" in x and x["tags"] is not None], [])))
    all_tags = sorted(all_tags)
    save_new_tags(all_tags)
    save_new_entries(all_entries)


if __name__ == "__main__":
    with SetupClient():
        main()
