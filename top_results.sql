WITH nearest_neighbors AS (
    SELECT
        a.id AS source_article_id,
        b.id AS neighbor_article_id,
        b.score AS neighbor_score,
        a.score AS source_score,
        a.embedding <-> b.embedding AS distance,
        a.agent AS source_agent
    FROM
        article a
    CROSS JOIN
        article b
    WHERE
        a.created_at >= NOW() - INTERVAL '2 days'
        AND a.id != b.id  -- Exclude the article itself from its neighbors
    ORDER BY
        distance
    LIMIT 10  -- Limit to 10 nearest neighbors for each article
), scored_articles AS (
    SELECT
        source_article_id,
        source_score,
        CASE
            WHEN source_agent = 'user' THEN source_score
            ELSE AVG(neighbor_score)
        END AS calculated_score
    FROM
        nearest_neighbors
    GROUP BY
        source_article_id, source_score, source_agent
)
SELECT
    a.*,
    sa.calculated_score AS neighbor_score
FROM
    article a
JOIN
    scored_articles sa ON a.id = sa.source_article_id
ORDER BY
    neighbor_score DESC;
