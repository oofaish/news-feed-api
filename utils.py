import os

from typing import Optional
from urllib.parse import urlparse, urlunparse

from bs4 import BeautifulSoup
from supabase import Client, create_client


ARTICLE_TABLE = "article"
TAG_TABLE = "tag"
FEED_TABLE = "feed"
MODEL_TABLE = "model"

bad_stuff = ["&raquo;", "&amp;"]


def remove_query_string(uri):
    parsed_uri = urlparse(uri)
    return urlunparse((parsed_uri.scheme, parsed_uri.netloc, parsed_uri.path, None, None, None))


def extract_text_from_p_tags(html: str) -> str:
    """
    Useful for the Guardian summary.
    """
    soup = BeautifulSoup(html, "html.parser")
    p_tags = soup.find_all("p")
    extracted_texts = [p.text for p in p_tags]
    return ". ".join(extracted_texts)


def chunk_list(lst, chunk_size):
    """Yield successive chunk_size chunks from lst."""
    for i in range(0, len(lst), chunk_size):
        yield lst[i : i + chunk_size]


authed_client: Optional[Client] = None


def get_authenticated_client():
    global authed_client
    if authed_client is None:
        raise ValueError("must use setup_client decorator")
    return authed_client


def _get_authenticated_client():
    global authed_client
    # authed_client = None
    if authed_client is None:
        authed_client = create_client(
            os.environ.get("SUPABASE_URL"),
            os.environ.get("SUPBASE_PROJECT_API_KEY"),
        )

        authed_client.auth.sign_in_with_password(
            {
                "email": os.environ.get("SUPABASE_ADMIN_USER"),
                "password": os.environ.get("SUPABASE_ADMIN_PASSWORD"),
            }
        )

    return authed_client


def _sign_out_authenticated_client():
    """
    required to avoid hanging!
    """
    global authed_client

    if authed_client is not None:
        authed_client.auth.sign_out()
        authed_client = None


def setup_client(wrapped_function):
    def wrapper(*args, **kwargs):
        _get_authenticated_client()
        try:
            return wrapped_function(*args, **kwargs)
        finally:
            _sign_out_authenticated_client()

    return wrapper


class SetupClient:
    def __enter__(self):
        _get_authenticated_client()

    def __exit__(self, exc_type, exc_value, traceback):
        _sign_out_authenticated_client()
