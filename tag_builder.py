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


def analyze_content(
    article: str,
    tags: Tags,
    user_profile: dict,
) -> TaggingAndScoreResult:
    prompt = f"""
    "You are a news article tagger and interest analyzer. Your task is to assign relevant tags and calculate a predicted interest score based on the provided taxonomy and user preferences.

    TAXONOMY: {tags.get_formatted_tags()}
    USER PREFERENCES: {json.dumps(user_profile)}

    ARTICLE DETAILS:
    {article}

    Please analyze the article and respond in the following JSON format:

    {{
    "scope": ["tag1", "tag2"] or ["UNKNOWN"],
    "topic": ["tag1", "tag2"] or ["UNKNOWN"],
    "mood": ["tag1"] or ["UNKNOWN"],
    "score": n
    }}

    Rules:
        1. Each dimension can have up to 2 tags (except mood, which should have exactly 1)
        2. Use 'unknown' as an array value if you cannot determine tags for a dimension
        3. Tags must exactly match the taxonomy provided
        4. Return the tags in the order most relevant to least relevant for the article
        4. Use the the description provided for each tag to help you make your decision
        5. Consider both article content and user preferences in tagging
        6. Provide an interest score between -10 and 10 where:
            10: Extremely relevant to user interests and preferences
            5: Moderately interesting based on user profile
            0: Neutral relevance
            -5: Likely not of interest
            -10: Strongly misaligned with user interests
        8. You can use all integers between -10 to 10 for the interest score
        7. Provide no explanation or additional text - only the JSON response
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
        result = json.loads(response_text)
        return TaggingAndScoreResult(scope=result.get("scope", None), topic=result.get("topic", None), mood=result.get("mood", None), score=result.get("score", None))
    except json.JSONDecodeError as e:
        logger.exception("An unexpected error occurred")
        return TaggingAndScoreResult(scope=None, topic=None, mood=None, score=None, error=f"JSON Error: {str(e)}")
    except Exception as e:
        logger.exception("An unexpected error occurred")
        return TaggingAndScoreResult(scope=None, topic=None, mood=None, score=None, error=f"Unexpected Error: {str(e)}")


if __name__ == "__main__":
    user_profile = load_user_profile()
    tags = get_tags()

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

    result = analyze_content(full_text, tags, user_profile)

    print(result)

    pass
