CREATE table unique_fragments (
                                  fragment_id serial primary key,
                                  fragment text,
                                  fragment_embedding vector(1024)
);
CREATE INDEX ON unique_fragments USING hnsw (fragment_embedding vector_cosine_ops);

