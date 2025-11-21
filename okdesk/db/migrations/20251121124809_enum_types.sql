-- migrate:up

-- Create ENUM types for the support system

-- Source type: OKDesk or Telegram
CREATE TYPE source_type AS ENUM ('okdesk', 'telegram');

-- Issue status lifecycle
CREATE TYPE issue_status AS ENUM ('opened', 'wait', 'completed', 'closed');

-- Author type in messages
CREATE TYPE author_type AS ENUM ('employee', 'contact', 'user');

-- Import job status
CREATE TYPE import_status AS ENUM ('in_progress', 'completed', 'failed');

-- Tag origin/type
CREATE TYPE tag_type AS ENUM ('auto', 'okdesk', 'manual');

-- migrate:down

-- Drop ENUM types in reverse order
DROP TYPE IF EXISTS tag_type;
DROP TYPE IF EXISTS import_status;
DROP TYPE IF EXISTS author_type;
DROP TYPE IF EXISTS issue_status;
DROP TYPE IF EXISTS source_type;

