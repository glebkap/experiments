-- migrate:up

-- Enable PostgreSQL extensions required for the application

-- pgcrypto extension for gen_random_uuid() function
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- migrate:down

-- Drop the extension
DROP EXTENSION IF EXISTS pgcrypto;
