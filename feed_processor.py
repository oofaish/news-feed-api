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
    try:
        feed = feedparser.parse(feed_url)
        logger.info(f"Successfully fetched feed from {feed_url} with {len(feed.entries)} entries")
        return feed
    except Exception as e:
        logger.error(f"Failed to fetch feed from {feed_url}: {e}")
        raise


def process_feed(publication: Publication, feed: feedparser.FeedParserDict) -> Iterator[dict[str, Any]]:
    total_entries = len(feed.entries)
    processed_count = 0
    skipped_count = 0
    error_count = 0

    logger.info(f"Processing {total_entries} entries from {publication.value}")
    cutoff_date = (datetime.datetime.now() - datetime.timedelta(days=MAX_AGE_FOR_ARTICLE_FOR_PARSSING)).timetuple()

    for entry in feed.entries:
        # dont bother with articles more than 5 days old
        if entry["published_parsed"] > cutoff_date:
            try:
                processed_count += 1
                yield process(publication, entry)
            except Exception as e:
                error_count += 1
                logger.error(f"Error processing {entry.get('link', 'unknown link')} from {publication.value}: {e}")
        else:
            skipped_count += 1

    logger.info(f"Feed processing summary for {publication.value}: {processed_count} processed, {skipped_count} skipped (too old), {error_count} errors")


def get_all_feed_urls() -> list[tuple[Publication, list[str]]]:
    # load the yaml file of publications.yaml
    try:
        with open("publications.yaml", "r") as f:
            publications = yaml.safe_load(f)

        result = [(Publication(p["publication"]), p["urls"]) for p in publications if p["enabled"]]
        enabled_count = len(result)
        total_count = len(publications)
        logger.info(f"Loaded {enabled_count} enabled publications out of {total_count} total")

        return result
    except Exception as e:
        logger.error(f"Failed to load publications: {e}")
        raise
    # client = get_authenticated_client()
    # active_feeds = client.table(FEED_TABLE).select("title,url").eq("enabled", True).execute()
    # feeds = [(row["title"], row["url"]) for row in active_feeds.data]
    # return feeds


def process_one_feed(url: str, publication: Publication, seen_links: set[str]):
    logger.info(f"Processing feed from {url} for publication {publication.value}")
    try:
        feed = get_feed(url)
        entries = process_feed(publication, feed)
        all_entries = []
        duplicates_count = 0
        bad_content_count = 0
        html_errors = 0

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

            if is_bad:
                bad_content_count += 1
                continue

            if entry["link"] in seen_links:
                duplicates_count += 1
                continue

            seen_links.add(entry["link"])
            try:
                entry["summary"] = remove_html(entry["summary"])
                all_entries.append(entry)
            except Exception as e:
                html_errors += 1
                logger.exception(f"Failed to remove html from summary for {entry.get('link', 'unknown link')}: {e}")

        logger.info(f"Processed feed from {url}: {len(all_entries)} valid entries, {duplicates_count} duplicates, {bad_content_count} filtered, {html_errors} HTML errors")
        return all_entries
    except Exception as e:
        logger.error(f"Failed to process feed from {url} for {publication.value}: {e}")
        return []


def process_all_feeds():
    all_entries = []
    all_feed_urls = get_all_feed_urls()
    publication: Publication
    seen_entry_urls = set()
    total_publications = len(all_feed_urls)
    total_feeds = sum(len(urls) for _, urls in all_feed_urls)

    logger.info(f"Starting to process {total_feeds} feeds from {total_publications} publications")

    publication_count = 0
    for publication, feed_urls in all_feed_urls:
        publication_count += 1
        publication_entries = []
        logger.info(f"Processing publication {publication_count}/{total_publications}: {publication.value} with {len(feed_urls)} feeds")

        for url in feed_urls:
            entries = process_one_feed(url, publication, seen_entry_urls)
            publication_entries.extend(entries)

        logger.info(f"Collected {len(publication_entries)} entries from {publication.value}")
        all_entries.extend(publication_entries)

    logger.info(f"Completed processing all feeds: collected {len(all_entries)} total entries")
    return all_entries


def save_new_entries_hacker_rank(entries: list[dict[str, Any]]):
    if not entries:
        logger.info("No Hacker News entries to save")
        return

    logger.info(f"Saving {len(entries)} Hacker News entries to database with Summary and title updates")
    client = get_authenticated_client()
    so_far = 0
    chunk_count = 0
    total_chunks = len(list(chunk_list(entries, 50)))

    try:
        for chunk in chunk_list(entries, 50):
            chunk_count += 1
            chunk_size = len(chunk)
            so_far += chunk_size
            logger.info(f"Saving Hacker News chunk {chunk_count}/{total_chunks}: {so_far}/{len(entries)} entries")

            try:
                # Update Summary and title on conflict, leave other columns as is
                client.table(ARTICLE_TABLE).upsert(chunk, on_conflict="link", merge=["summary", "title"]).execute()
                logger.info(f"Successfully saved Hacker News chunk {chunk_count}: {chunk_size} entries")
            except Exception as e:
                logger.error(f"Failed to save Hacker News chunk {chunk_count}: {e}")

        logger.info(f"Completed saving {so_far} Hacker News entries to database")
    except Exception as e:
        logger.error(f"Error during Hacker News save operation: {e}")


def save_new_entries(entries: list[dict[str, Any]]):
    if not entries:
        logger.info("No new entries to save")
        return

    logger.info(f"Saving {len(entries)} new entries to database")
    client = get_authenticated_client()
    so_far = 0
    chunk_count = 0
    total_chunks = len(list(chunk_list(entries, 50)))

    try:
        for chunk in chunk_list(entries, 50):
            chunk_count += 1
            chunk_size = len(chunk)
            so_far += chunk_size
            logger.info(f"Saving chunk {chunk_count}/{total_chunks}: {so_far}/{len(entries)} entries")

            try:
                client.table(ARTICLE_TABLE).upsert(chunk, on_conflict="link (DO NOTHING)").execute()
                logger.info(f"Successfully saved chunk {chunk_count}: {chunk_size} entries")
            except Exception as e:
                logger.error(f"Failed to save chunk {chunk_count}: {e}")

        logger.info(f"Completed saving {so_far} entries to database")
    except Exception as e:
        logger.error(f"Error during save operation: {e}")


def main():
    logger.info("Starting feed processor main function")
    try:
        start_time = datetime.datetime.now()
        all_entries = process_all_feeds()
        processing_time = datetime.datetime.now() - start_time

        logger.info(f"Feed processing completed in {processing_time.total_seconds():.2f} seconds, found {len(all_entries)} entries")

        # Filter entries by publication
        hacker_news_entries = [entry for entry in all_entries if entry.get("publication") == Publication.HackerNews.value]
        other_entries = [entry for entry in all_entries if entry.get("publication") != Publication.HackerNews.value]

        logger.info(f"Filtered {len(hacker_news_entries)} Hacker News entries and {len(other_entries)} other entries")

        save_start_time = datetime.datetime.now()

        # Call the appropriate save function based on publication
        save_new_entries_hacker_rank(hacker_news_entries)
        save_new_entries(other_entries)

        save_time = datetime.datetime.now() - save_start_time

        logger.info(f"Saving entries completed in {save_time.total_seconds():.2f} seconds")
        logger.info(f"Total feed processor execution time: {(datetime.datetime.now() - start_time).total_seconds():.2f} seconds")
    except Exception as e:
        logger.exception(f"Error in feed processor main function: {e}")


if __name__ == "__main__":
    # feeds = get_all_feed_urls()
    main()
    pass
    # with SetupClient():
    #     # main()
    #     # result = process_one_feed("https://hnrss.org/show?points=100&comments=25", "Hacker News")
    #     result = process_one_feed("https://feeds.arstechnica.com/arstechnica/index", "Ars Technica")
    #     pass
