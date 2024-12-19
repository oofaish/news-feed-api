from logging import getLogger
from typing import Any

from config import Publication
from utils import extract_text_from_p_tags, remove_query_string


logger = getLogger(__name__)


def wsj_and_ft_parser(entry: dict[str, Any]) -> dict[str, Any]:
    link = remove_query_string(entry["link"])

    if "ft.com" in link:
        publication = "FT"
        if "tags" in entry and len(entry["tags"]):
            logger.info(f"hmm - FT had {entry['tags']} for tags")

    elif "wsj.com" in link or "wsj_articletype" in entry:
        if "tags" in entry and len(entry["tags"]) > 2:
            raise ValueError(f"hmm - FT had {entry['tags']} for tags")

        publication = "WSJ"
    else:
        raise ValueError(f"Unexpected link {link}")

    return {
        "title": entry["title"],
        "link": link,
        "publication": publication,
        "summary": entry.get("summary"),
        "published_at": entry["published"],
        "tags": set(),
    }


def guardian_and_nyt_parser(entry: dict[str, Any]) -> dict[str, Any]:
    if "nytimes.com" in entry["link"]:
        publication = "NY Times"
        summary = entry.get("summary")
    elif "guardian." in entry["link"]:
        publication = "Guardian"
        if "summary" in entry:
            summary = extract_text_from_p_tags(entry["summary"])
        else:
            summary = None
    else:
        raise ValueError(f"Unexpected link {entry['link']}")

    # if "media_content" in entry and len(entry["media_content"]) and "url" in entry["media_content"][0]:
    #     media = entry["media_content"][0]["url"]
    # else:
    media = None

    return {
        "title": entry["title"],
        "link": remove_query_string(entry["link"]),
        "publication": publication,
        "summary": summary,
        "tags": set(),  # [publication],
        "published_at": entry["published"],
        "media": media,
        "author": entry.get("author"),
    }


def hnrss_parser(entry: dict[str, Any]) -> dict[str, Any]:
    summary = None
    if "summary" in entry:
        summary = extract_text_from_p_tags(entry["summary"])

    author = None
    if "author" in entry:
        author = entry["author"]
    elif "authors" in entry and entry["authors"]:
        author = entry["authors"][0].get("name")

    return {
        "title": entry["title"],
        "link": entry["comments"],
        "publication": "Hacker News",
        "summary": summary,
        "tags": set(),  # ["Hacker News"],
        "published_at": entry["published"],
        "media": None,
        "author": author,
    }


def default_parser(entry: dict[str, Any]) -> dict[str, Any]:
    summary = None
    if "summary" in entry:
        summary = extract_text_from_p_tags(entry["summary"])

    author = None
    if "author" in entry:
        author = entry["author"]
    elif "authors" in entry and entry["authors"]:
        author = entry["authors"][0].get("name")

    publication = None
    if "source" in entry and "title" in entry["source"]:
        publication = entry["source"]["title"]

    return {
        "title": entry["title"],
        "link": entry["link"],
        "publication": publication,
        "summary": summary,
        "tags": set(),
        "published_at": entry["published"],
        "media": None,
        "author": author,
    }


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
    if "wsj.com" in entry["link"] or "ft.com" in entry["link"] or "wsj_articletype" in entry:
        result = wsj_and_ft_parser(entry)
    elif "guardian." in entry["link"] or "nytimes.com" in entry["link"]:
        result = guardian_and_nyt_parser(entry)
    elif "news.ycombinator.com" in entry.get("id", ""):
        result = hnrss_parser(entry)
    else:
        result = default_parser(entry)

    if result["tags"] is None:
        result["tags"] = {title.value}
    else:
        result["tags"].add(title.value)

    return ensure_fields(result)
