-- migrate:up
-- Удаление устаревших таблиц для анализа сообщений
-- Эти таблицы заменяются на новую архитектуру с preprocessing и clustering

DROP TABLE IF EXISTS message_intents CASCADE;
DROP TABLE IF EXISTS message_tags CASCADE;
DROP TABLE IF EXISTS message_analysis CASCADE;
DROP TABLE IF EXISTS intents CASCADE;
DROP TABLE IF EXISTS tags CASCADE;

-- migrate:down
-- Восстановление таблиц (если потребуется откат)

CREATE TYPE tag_type AS ENUM ('auto', 'okdesk', 'manual');

CREATE TABLE intents (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  code TEXT NOT NULL UNIQUE,
  name TEXT NOT NULL,
  description TEXT NOT NULL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE message_analysis (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  message_id UUID REFERENCES messages(id) UNIQUE,
  analyzed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  reasoning TEXT
);

CREATE TABLE message_intents (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  message_analysis_id UUID REFERENCES message_analysis(id),
  intent_id UUID REFERENCES intents(id),
  confidence FLOAT CHECK (confidence >= 0 AND confidence <= 1),
  UNIQUE(message_analysis_id, intent_id)
);

CREATE TABLE tags (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  name TEXT NOT NULL UNIQUE,
  type tag_type,
  source VARCHAR(50),
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE message_tags (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  message_analysis_id UUID REFERENCES message_analysis(id),
  tag_id UUID REFERENCES tags(id),
  confidence FLOAT CHECK (confidence >= 0 AND confidence <= 1),
  UNIQUE(message_analysis_id, tag_id)
);
