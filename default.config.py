# this was the old model
# model="text-embedding-ada-002"
from enum import Enum


EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_COMPLETION_MODEL = "gpt-4o-mini"

# recent article to try and get embeddings for
MAX_RECENT_ARTICLES = 200
RECENT_HOURS = 24

# max size of title and content we send for embedding
MAX_TITLE_SIZE = 1000
MAX_SIZE = 10000

# length of embedding
EMBEDDING_LENGTH = 1536

MAX_AGE_FOR_ARTICLE_FOR_PARSSING = 5


class Publication(Enum):
    pass
