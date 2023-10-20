CREATE OR REPLACE FUNCTION update_articles_with_agent_results(data jsonb)
RETURNS boolean LANGUAGE plpgsql AS $$
DECLARE
    row_data jsonb;
BEGIN
    FOR row_data IN SELECT * FROM jsonb_array_elements(data)
    LOOP
        UPDATE article
        SET agent = row_data->>'agent',
            score = (row_data->>'score')::int,
            updated_at = now(),
            reason = row_data->>'reason'
        WHERE id = (row_data->>'id')::int;
    END LOOP;
    RETURN true;
END;
$$;
