import json
import logging
import os

from dataclasses import dataclass

import yaml

from openai import OpenAI

from config import CHAT_COMPLETION_MODEL


logger = logging.getLogger(__name__)


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


@dataclass
class Tag:
    tag: str
    description: str


# TODO lols tag dimensions should really be dynamic


@dataclass
class Tags:
    scope: list[Tag]
    topic: list[Tag]
    mood: list[Tag]

    def get_formatted_tags(self):
        result = "\n".join([f"{dimension}:\n" + "\n".join([f"- {tag.tag}: {tag.description}" for tag in getattr(self, dimension)]) for dimension in ["scope", "topic", "mood"]])
        return result


def get_tags() -> Tags:
    with open("tags.yaml", "r") as file:
        tag_data = yaml.safe_load(file)

    return Tags(
        scope=[Tag(tag=k, description=v) for k, v in tag_data.get("scope", {}).items()],
        topic=[Tag(tag=k, description=v) for k, v in tag_data.get("topic", {}).items()],
        mood=[Tag(tag=k, description=v) for k, v in tag_data.get("mood", {}).items()],
    )


def load_user_profile():
    with open("user_profile.json", "r") as file:
        return json.load(file)


@dataclass
class TaggingAndScoreResult:
    scope: list[str] | None
    topic: list[str] | None
    mood: list[str] | None
    score: int | None
    error: str | None = None


def parse_answer(response_text: str) -> dict:
    if not response_text:
        raise json.JSONDecodeError("Empty input", "", 0)
    if "```" in response_text:
        parts = response_text.split("```")
        if len(parts) >= 3:
            response_text = parts[1]
            if response_text.startswith("json"):
                response_text = response_text[4:]
    cleaned_text = (
        response_text.strip()  # Remove leading/trailing whitespace
        .strip("\"'")  # Remove surrounding quotes
        .encode()
        .decode("unicode_escape")  # Handle escape characters
        .replace("'", '"')  # Replace single quotes with double quotes
        .replace("\n", "")  # Remove newlines
    )
    return json.loads(cleaned_text)


def analyze_content(
    article: str,
    tags: Tags | None = None,
    user_profile: dict | None = None,
) -> TaggingAndScoreResult:
    if tags is None:
        tags = get_tags()
    if user_profile is None:
        user_profile = load_user_profile()

    prompt = f"""
    You are a precise content analyzer focused on personalized news filtering. Analyze this article based on the user's interests and preferences.

    TAXONOMY: {tags.get_formatted_tags()}
    USER PREFERENCES: {json.dumps(user_profile)}

    ARTICLE:
    {article}

    Respond in JSON format:
    {{
        "scope": ["tag1", "tag2"] or ["UNKNOWN"],
        "topic": ["tag1", "tag2"] or ["UNKNOWN"],
        "mood": ["tag1"] or ["UNKNOWN"],
        "score": n
    }}

    Scoring Guidelines:
    1. Start with a base score of 0
    2. Add points:
       - +3 to +4 for matching personal interests
       - +1 to +2 for preferred content types
       - +1 to +2 for matching professional interests
       - +1 for matching tone and depth preferences
    3. Subtract points:
       - -3 to -5 for explicitly avoided topics
       - -2 for avoided content types
       - -1 for mismatched tone or depth
    4. Final score must be between -10 and 10

    Rules:
    1. Each dimension can have up to 2 tags (except mood: exactly 1).
    2. Only allocate a tag if the relevance is high. Preference is to have UNKNOWN tag or just 1 tag rather than a tag with low relevance.
    3. Use 'UNKNOWN' if tags cannot be determined
    4. Tags must exactly match the provided taxonomy
    5. Order tags by relevance
    6. Provide no explanation - only JSON response
    7. Be especially strict about negative scoring for sports content
    """

    try:
        response = client.chat.completions.create(
            model=CHAT_COMPLETION_MODEL,
            messages=[
                # {"role": "system", "content": "You are a precise content analyzer. Respond only with the requested format."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,  # Lower temperature for more consistent outputs
        )

        # Parse the response and extract tags, suggestions, and rating
        response_text = response.choices[0].message.content

    except Exception as e:
        logger.exception("An unexpected error occurred")
        return TaggingAndScoreResult(scope=None, topic=None, mood=None, score=None, error=f"LLM Error: {str(e)}")

    try:
        result = parse_answer(response_text)
        return TaggingAndScoreResult(scope=result.get("scope", None), topic=result.get("topic", None), mood=result.get("mood", None), score=result.get("score", None))
    except json.JSONDecodeError as e:
        logger.exception("An unexpected error occurred")
        return TaggingAndScoreResult(scope=None, topic=None, mood=None, score=None, error=f"JSON Error: {str(e)}")
    except Exception as e:
        logger.exception("An unexpected error occurred")
        return TaggingAndScoreResult(scope=None, topic=None, mood=None, score=None, error=f"Unexpected Error: {str(e)}")


if __name__ == "__main__":
    title = "South Korea deploys K-pop light sticks and dance in protests against president"
    summary = """Christmas carols, K-pop merchandise and food trucks create positive atmosphere as protesters seek to oust President Yoon over his martial law attempt.
    With blasting K-pop, glow sticks, food trucks and obligatory selfies, the protests that have swelled across South Korea since the president’s shock declaration of
    martial law last week have taken on a surprisingly festive mood.. Outside the national assembly in Seoul on Tuesday night, food trucks lined the streets selling
    traditional Korean snacks like tteokbokki (spicy rice cakes), sundae (blood sausage), and even beondegi, the favourite winter treat of boiled silkworm pupae.
    """

    publication = "Guardian"

    full_text = f"""
        News article fom {publication}

        Title: {title}

        Summary: {summary}
    """

    result = analyze_content(full_text)

    print(result)

    pass
