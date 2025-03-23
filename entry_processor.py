from logging import getLogger
from typing import Any

from config import Publication
from utils import extract_text_from_p_tags, remove_query_string


logger = getLogger(__name__)


def wsj_and_ft_parser(entry: dict[str, Any]) -> dict[str, Any]:
    link = remove_query_string(entry["link"])
    source = "Financial Times" if "ft.com" in link else "Wall Street Journal"
    logger.debug(f"Processing {source} article: {entry.get('title', 'No title')}")

    if "ft.com" in link:
        if "tags" in entry and len(entry["tags"]):
            logger.info(f"FT article has tags: {entry['tags']}")

    elif "wsj.com" in link or "wsj_articletype" in entry:
        if "tags" in entry and len(entry["tags"]) > 2:
            logger.warning(f"WSJ article has unexpected number of tags: {entry['tags']}")
            raise ValueError(f"WSJ article has too many tags: {entry['tags']}")
    else:
        logger.error(f"Unexpected link in wsj_and_ft_parser: {link}")
        raise ValueError(f"Unexpected link {link}")

    logger.debug(f"Successfully parsed {source} article: {entry.get('title', 'No title')}")
    return {
        "title": entry["title"],
        "link": link,
        "summary": entry.get("summary"),
        "published_at": entry["published"],
        "tags": set(),
    }


def guardian_and_nyt_parser(entry: dict[str, Any]) -> dict[str, Any]:
    link = entry["link"]
    source = "New York Times" if "nytimes.com" in link else "The Guardian"
    logger.debug(f"Processing {source} article: {entry.get('title', 'No title')}")

    try:
        if "nytimes.com" in link:
            summary = entry.get("summary")
            logger.debug(f"NYT article summary length: {len(summary) if summary else 0} chars")
        elif "guardian." in link:
            if "summary" in entry:
                summary = extract_text_from_p_tags(entry["summary"])
                logger.debug(f"Guardian article summary extracted, length: {len(summary) if summary else 0} chars")
            else:
                logger.debug("Guardian article has no summary")
                summary = None
        else:
            logger.error(f"Unexpected link in guardian_and_nyt_parser: {link}")
            raise ValueError(f"Unexpected link {link}")

        # if "media_content" in entry and len(entry["media_content"]) and "url" in entry["media_content"][0]:
        #     media = entry["media_content"][0]["url"]
        # else:
        media = None

        has_author = "author" in entry and entry["author"] is not None
        logger.debug(f"Successfully parsed {source} article: {entry.get('title', 'No title')}, has author: {has_author}")

        return {
            "title": entry["title"],
            "link": remove_query_string(link),
            "summary": summary,
            "tags": set(),  # [publication],
            "published_at": entry["published"],
            "media": media,
            "author": entry.get("author"),
        }
    except Exception as e:
        logger.exception(f"Error parsing {source} article: {e}")
        raise


def hnrss_parser(entry: dict[str, Any]) -> dict[str, Any]:
    logger.debug(f"Processing Hacker News article: {entry.get('title', 'No title')}")

    try:
        summary = None
        if "summary" in entry:
            raw_summary = extract_text_from_p_tags(entry["summary"])
            logger.debug(f"HN article raw summary length: {len(raw_summary) if raw_summary else 0} chars")
            try:
                # get rid of the links that this is putting in summary
                summary = f"Points:{raw_summary.split('Points:')[-1]}"
                logger.debug("Extracted points info from HN summary")
            except Exception as e:
                logger.exception(f"Error extracting points from HN summary: {e}")
                summary = raw_summary

        author = None
        if "author" in entry:
            author = entry["author"]
            logger.debug(f"HN article has author: {author}")
        elif "authors" in entry and entry["authors"]:
            author = entry["authors"][0].get("name")
            logger.debug(f"HN article has author from authors list: {author}")
        else:
            logger.debug("HN article has no author information")

        logger.debug(f"Successfully parsed HN article: {entry.get('title', 'No title')}")
        return {
            "title": entry["title"],
            "link": entry["comments"],
            "publication": Publication.HackerNews,
            "summary": summary,
            "tags": set(),  # ["Hacker News"],
            "published_at": entry["published"],
            "media": None,
            "author": author,
        }
    except Exception as e:
        logger.exception(f"Error parsing Hacker News article: {e}")
        raise


def default_parser(entry: dict[str, Any]) -> dict[str, Any]:
    source = entry.get("source", {}).get("title", "Unknown source")
    logger.debug(f"Processing article from {source}: {entry.get('title', 'No title')}")

    try:
        summary = None
        if "summary" in entry:
            summary = extract_text_from_p_tags(entry["summary"])
            logger.debug(f"Article summary extracted, length: {len(summary) if summary else 0} chars")
        else:
            logger.debug("Article has no summary")

        author = None
        if "author" in entry:
            author = entry["author"]
            logger.debug(f"Article has author: {author}")
        elif "authors" in entry and entry["authors"]:
            author = entry["authors"][0].get("name")
            logger.debug(f"Article has author from authors list: {author}")
        else:
            logger.debug("Article has no author information")

        logger.debug(f"Successfully parsed article from {source}: {entry.get('title', 'No title')}")
        return {
            "title": entry["title"],
            "link": entry["link"],
            "summary": summary,
            "tags": set(),
            "published_at": entry["published"],
            "media": None,
            "author": author,
        }
    except Exception as e:
        logger.exception(f"Error parsing article from {source}: {e}")
        raise


def ensure_fields(entry: dict[str, Any]) -> dict[str, Any]:
    required_format = {
        "summary": None,
        "tags": None,
        "media": None,
        "author": None,
    }
    required_format.update(entry)

    return required_format


def process(title: Publication, entry: dict[str, Any]) -> dict[str, Any]:
    entry_id = entry.get("id", "No ID")
    entry_link = entry.get("link", "No link")
    entry_title = entry.get("title", "No title")

    logger.debug(f"Processing entry from {title.value}: '{entry_title}' (ID: {entry_id})")

    try:
        if "news.ycombinator.com" in entry.get("id", ""):
            logger.info(f"Processing Hacker News entry: '{entry_title}'")
            result = hnrss_parser(entry)
        elif "wsj.com" in entry_link or "ft.com" in entry_link or "wsj_articletype" in entry:
            source = "Financial Times" if "ft.com" in entry_link else "Wall Street Journal"
            logger.info(f"Processing {source} entry: '{entry_title}'")
            result = wsj_and_ft_parser(entry)
        elif "guardian." in entry_link or "nytimes.com" in entry_link:
            source = "New York Times" if "nytimes.com" in entry_link else "The Guardian"
            logger.info(f"Processing {source} entry: '{entry_title}'")
            result = guardian_and_nyt_parser(entry)
        else:
            logger.info(f"Processing entry using default parser: '{entry_title}' from {title.value}")
            result = default_parser(entry)

        if result["tags"] is None:
            result["tags"] = {title.value}
            logger.debug(f"Setting tags to publication: {title.value}")
        else:
            result["tags"].add(title.value)
            logger.debug(f"Adding publication to tags: {title.value}")

        final_result = ensure_fields(result)
        logger.info(f"Successfully processed entry: '{entry_title}' from {title.value}")
        return final_result
    except Exception as e:
        logger.exception(f"Error processing entry '{entry_title}' from {title.value}: {e}")
        raise
