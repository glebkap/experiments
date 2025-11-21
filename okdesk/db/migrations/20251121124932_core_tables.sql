-- migrate:up

-- Create core tables for the support system

-- 1. Sources table: Configuration for data sources (OKDesk, Telegram)
CREATE TABLE sources (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  name TEXT NOT NULL,
  type source_type NOT NULL,
  config JSONB,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 2. Issues table: Support tickets/issues
CREATE TABLE issues (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  external_id VARCHAR(255) NOT NULL,
  source_id UUID REFERENCES sources(id),
  title TEXT,
  description TEXT,
  status issue_status,
  priority INTEGER CHECK (priority BETWEEN 1 AND 4),
  created_at TIMESTAMP,
  updated_at TIMESTAMP,
  completed_at TIMESTAMP,
  UNIQUE(external_id, source_id)
);

-- 3. Messages table: Individual messages within issues
CREATE TABLE messages (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  issue_id UUID REFERENCES issues(id),
  external_id VARCHAR(255) NOT NULL,
  author_id VARCHAR(255),
  author_name TEXT,
  author_type author_type,
  content TEXT NOT NULL,
  is_public BOOLEAN DEFAULT true,
  published_at TIMESTAMP,
  UNIQUE(external_id, issue_id)
);

-- 4. Intents table: Dynamic catalog of user intentions (populated by LLM)
CREATE TABLE intents (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  code TEXT NOT NULL UNIQUE,
  name TEXT NOT NULL,
  description TEXT NOT NULL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 5. Message Analysis table: Analysis results for messages
CREATE TABLE message_analysis (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  message_id UUID REFERENCES messages(id) UNIQUE,
  analyzed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  reasoning TEXT
);

-- 6. Message Intents table: Many-to-many relationship between messages and intents
CREATE TABLE message_intents (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  message_analysis_id UUID REFERENCES message_analysis(id),
  intent_id UUID REFERENCES intents(id),
  confidence FLOAT CHECK (confidence >= 0 AND confidence <= 1),
  UNIQUE(message_analysis_id, intent_id)
);

-- 7. Tags table: Auto-generated, OKDesk, or manual tags
CREATE TABLE tags (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  name TEXT NOT NULL UNIQUE,
  type tag_type,
  source VARCHAR(50),
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 8. Message Tags table: Message-to-tag relationships
CREATE TABLE message_tags (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  message_analysis_id UUID REFERENCES message_analysis(id),
  tag_id UUID REFERENCES tags(id),
  confidence FLOAT CHECK (confidence >= 0 AND confidence <= 1),
  UNIQUE(message_analysis_id, tag_id)
);

-- 9. Intent Clusters table: Grouping of similar intents
CREATE TABLE intent_clusters (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  name TEXT NOT NULL,
  description TEXT,
  pattern TEXT,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 10. Message Clusters table: Message-to-cluster assignments
CREATE TABLE message_clusters (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  message_id UUID REFERENCES messages(id),
  cluster_id UUID REFERENCES intent_clusters(id),
  similarity_score FLOAT CHECK (similarity_score >= 0 AND similarity_score <= 1),
  UNIQUE(message_id, cluster_id)
);

-- 11. Imports table: Import history and statistics
CREATE TABLE imports (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  source_id UUID REFERENCES sources(id),
  filename TEXT,
  file_path TEXT,
  started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  completed_at TIMESTAMP,
  status import_status,
  stats JSONB,
  error_message TEXT
);

-- migrate:down

-- Drop tables in reverse order to respect foreign key constraints
DROP TABLE IF EXISTS imports;
DROP TABLE IF EXISTS message_clusters;
DROP TABLE IF EXISTS intent_clusters;
DROP TABLE IF EXISTS message_tags;
DROP TABLE IF EXISTS tags;
DROP TABLE IF EXISTS message_intents;
DROP TABLE IF EXISTS message_analysis;
DROP TABLE IF EXISTS intents;
DROP TABLE IF EXISTS messages;
DROP TABLE IF EXISTS issues;
DROP TABLE IF EXISTS sources;
