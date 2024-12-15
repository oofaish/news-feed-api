from logging import getLogger

import openai

from tenacity import retry, stop_after_attempt, wait_random_exponential


logger = getLogger(__name__)


@retry(wait=wait_random_exponential(min=5, max=20), stop=stop_after_attempt(6))
def get_embedding(text: str, model, **kwargs) -> list[float]:
    # replace newlines, which can negatively affect performance.
    text = text.replace("\n", " ")

    response = openai.embeddings.create(input=[text], model=model, **kwargs)

    return response.data[0].embedding


@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
def get_embeddings(list_of_text: list[str], model, **kwargs) -> list[list[float]]:
    assert len(list_of_text) <= 2048, "The batch size should not be larger than 2048."

    # replace newlines, which can negatively affect performance.
    list_of_text = [text.replace("\n", " ") for text in list_of_text]

    data = openai.embeddings.create(input=list_of_text, model=model, **kwargs).data
    return [d.embedding for d in data]
