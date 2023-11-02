from typing import Any, Optional

from utils import extract_text_from_p_tags, remove_query_string


def wsj_and_ft_parser(entry: dict[str, Any]) -> dict[str, Any]:
    if "wsj_articletype" in entry:
        tags = [entry["wsj_articletype"]]
    else:
        tags = None

    link = remove_query_string(entry["link"])

    if "ft.com" in link:
        publication = "FT"
        if "tags" in entry and len(entry["tags"]):
            print(f"hmm - FT had {entry['tags']} for tags")

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
        "tags": tags,
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

    if tags := entry.get("tags"):
        tags = [x["term"].title() for x in tags]

    if "media_content" in entry and len(entry["media_content"]) and "url" in entry["media_content"][0]:
        media = entry["media_content"][0]["url"]
    else:
        media = None

    return {
        "title": entry["title"],
        "link": remove_query_string(entry["link"]),
        "publication": publication,
        "summary": summary,
        "tags": tags,
        "published_at": entry["published"],
        "media": media,
        "author": entry.get("author"),
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


def process(title: Optional[str], entry: dict[str, Any]) -> dict[str, Any]:
    if "wsj.com" in entry["link"] or "ft.com" in entry["link"] or "wsj_articletype" in entry:
        result = wsj_and_ft_parser(entry)
    elif "guardian." in entry["link"] or "nytimes.com" in entry["link"]:
        result = guardian_and_nyt_parser(entry)
    else:
        raise ValueError(f"Unexpected link {entry['link']}")

    # TODO add the title to the result

    return ensure_fields(result)
