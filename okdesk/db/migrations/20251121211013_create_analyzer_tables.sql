-- migrate:up
-- Создание таблиц для Analyzer Service
-- Pipeline processing: preprocessing, embeddings, clustering

-- Таблица предобработанных issues (результат Stage 1)
CREATE TABLE preprocessed_issues (
  id UUID PRIMARY KEY REFERENCES issues(id) ON DELETE CASCADE,
  content TEXT NOT NULL,
  processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_preprocessed_issues_processed ON preprocessed_issues(processed_at);
CREATE INDEX idx_preprocessed_issues_text ON preprocessed_issues
  USING gin(to_tsvector('russian', content));

COMMENT ON TABLE preprocessed_issues IS 'Предобработанный текст issues (title + description)';
COMMENT ON COLUMN preprocessed_issues.id IS 'Ссылка на issues.id';
COMMENT ON COLUMN preprocessed_issues.content IS 'Очищенный и нормализованный текст (HTML удален, лемматизация)';

-- Таблица кластеров похожих issues
CREATE TABLE clusters (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  cluster_label INT NOT NULL UNIQUE,
  name TEXT,
  description TEXT,
  centroid_embedding FLOAT[] NOT NULL,
  size INT DEFAULT 0 CHECK (size >= 0),
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_clusters_size ON clusters(size DESC);
CREATE INDEX idx_clusters_label ON clusters(cluster_label);

COMMENT ON TABLE clusters IS 'Кластеры похожих issues (результат кластеризации)';
COMMENT ON COLUMN clusters.cluster_label IS 'Метка кластера из алгоритма (-1 = outlier, 0+ = кластер)';
COMMENT ON COLUMN clusters.name IS 'Название кластера (опционально через LLM)';
COMMENT ON COLUMN clusters.centroid_embedding IS 'Центроид кластера (среднее embedding)';
COMMENT ON COLUMN clusters.size IS 'Количество messages в кластере';

-- Таблица связи messages с кластерами
CREATE TABLE message_clusters (
  message_id UUID REFERENCES messages(id) ON DELETE CASCADE,
  cluster_id UUID REFERENCES clusters(id) ON DELETE CASCADE,
  distance_to_centroid FLOAT CHECK (distance_to_centroid >= 0 AND distance_to_centroid <= 1),
  assigned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (message_id, cluster_id)
);

CREATE INDEX idx_message_clusters_cluster_id ON message_clusters(cluster_id);
CREATE INDEX idx_message_clusters_distance ON message_clusters(distance_to_centroid);

COMMENT ON TABLE message_clusters IS 'Назначение messages к кластерам';
COMMENT ON COLUMN message_clusters.distance_to_centroid IS 'Cosine distance до центроида (0=близко, 1=далеко)';

-- migrate:down
-- Удаление таблиц analyzer service (откат)

DROP TABLE IF EXISTS message_clusters CASCADE;
DROP TABLE IF EXISTS clusters CASCADE;
DROP TABLE IF EXISTS preprocessed_issues CASCADE;
