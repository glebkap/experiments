-- migrate:up

-- Enable PostgreSQL extensions for full-text search

-- pg_trgm extension for trigram-based text search
-- Allows fuzzy text matching and similarity searches
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Create GIN index on messages content for fast full-text search
CREATE INDEX idx_messages_content_trgm ON messages USING gin (content gin_trgm_ops);

-- migrate:down

-- Drop the index first
DROP INDEX IF EXISTS idx_messages_content_trgm;

-- Drop the extension
DROP EXTENSION IF EXISTS pg_trgm;

