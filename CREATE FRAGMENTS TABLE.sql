CREATE table unique_fragments (
                                  fragment_id serial primary key,
                                  fragment text,
                                  fragment_embedding vector(1024)
)