CREATE OR REPLACE FUNCTION match_documents (
    query_embedding vector(768),
    match_count int DEFAULT 5,
    filter jsonb DEFAULT '{}'
)
RETURNS TABLE (
    id uuid,
    content text,
    metadata jsonb,
    similarity float
)
LANGUAGE plpgsql
AS $$
DECLARE
    doc_uuid_array uuid[];
BEGIN
    IF query_embedding IS NULL THEN
        RAISE EXCEPTION 'query_embedding cannot be NULL';
    END IF;

    -- Parse and convert document_ids array from JSON to uuid[]
    IF NOT (filter ? 'document_ids')THEN
        RAISE EXCEPTION 'Missing document_ids array in filter JSON';
    END IF;

    doc_uuid_array:= (
        SELECT array_agg(elem::uuid)
        FROM jsonb_array_elements_text(filter->'document_ids') elem
    );

    IF doc_uuid_array IS NULL OR array_length(doc_uuid_array, 1) = 0 THEN
        RAISE EXCEPTION 'document_ids array is empty or invalid';
    END IF;

    -- Return matching rows using only document_ids
    RETURN QUERY
    SELECT 
        ed.id,
        ed.content,
        ed.metadata,
        1 - (ed.embedding <=> query_embedding) AS similarity
    FROM embedded_documents ed
    WHERE 
        ed.document_id = ANY(doc_uuid_array)
    ORDER BY ed.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;