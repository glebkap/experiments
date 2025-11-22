-- migrate:up
-- Исправление: message_clusters -> issue_clusters (issue-centric approach)

-- 1. Переименовать message_clusters в issue_clusters
ALTER TABLE message_clusters RENAME TO issue_clusters;

-- 2. Переименовать message_id в issue_id
ALTER TABLE issue_clusters RENAME COLUMN message_id TO issue_id;

-- 3. Обновить foreign key constraint
ALTER TABLE issue_clusters DROP CONSTRAINT message_clusters_message_id_fkey;
ALTER TABLE issue_clusters ADD CONSTRAINT issue_clusters_issue_id_fkey
  FOREIGN KEY (issue_id) REFERENCES issues(id) ON DELETE CASCADE;

-- 4. Переименовать индексы
DROP INDEX IF EXISTS idx_message_clusters_cluster_id;
DROP INDEX IF EXISTS idx_message_clusters_distance;

CREATE INDEX idx_issue_clusters_cluster_id ON issue_clusters(cluster_id);
CREATE INDEX idx_issue_clusters_distance ON issue_clusters(distance_to_centroid);

-- 5. Обновить комментарии
COMMENT ON TABLE issue_clusters IS 'Назначение issues к кластерам';
COMMENT ON COLUMN issue_clusters.issue_id IS 'Ссылка на issues.id';
COMMENT ON COLUMN issue_clusters.distance_to_centroid IS 'Cosine distance до центроида (0=близко, 1=далеко)';

-- 6. Обновить размер кластеров в комментарии
COMMENT ON COLUMN clusters.size IS 'Количество issues в кластере';

-- migrate:down
-- Откат: issue_clusters -> message_clusters

-- Переименовать обратно
ALTER TABLE issue_clusters RENAME TO message_clusters;
ALTER TABLE message_clusters RENAME COLUMN issue_id TO message_id;

ALTER TABLE message_clusters DROP CONSTRAINT issue_clusters_issue_id_fkey;
ALTER TABLE message_clusters ADD CONSTRAINT message_clusters_message_id_fkey
  FOREIGN KEY (message_id) REFERENCES messages(id) ON DELETE CASCADE;

DROP INDEX IF EXISTS idx_issue_clusters_cluster_id;
DROP INDEX IF EXISTS idx_issue_clusters_distance;

CREATE INDEX idx_message_clusters_cluster_id ON message_clusters(cluster_id);
CREATE INDEX idx_message_clusters_distance ON message_clusters(distance_to_centroid);

COMMENT ON TABLE message_clusters IS 'Назначение messages к кластерам';
COMMENT ON COLUMN message_clusters.message_id IS 'Ссылка на messages.id';
COMMENT ON COLUMN clusters.size IS 'Количество messages в кластере';
