-- migrate:up

-- Create indexes for performance optimization

-- Messages table indexes
CREATE INDEX idx_messages_issue_id ON messages(issue_id);
CREATE INDEX idx_messages_published_at ON messages(published_at);

-- Message analysis index
CREATE INDEX idx_message_analysis_message_id ON message_analysis(message_id);

-- Issues table indexes
CREATE INDEX idx_issues_external_id ON issues(external_id);
CREATE INDEX idx_issues_created_at ON issues(created_at);

-- Message intents index
CREATE INDEX idx_message_intents_intent_id ON message_intents(intent_id);

-- Message tags index
CREATE INDEX idx_message_tags_tag_id ON message_tags(tag_id);

-- Intents index
CREATE INDEX idx_intents_code ON intents(code);

-- migrate:down

-- Drop indexes in reverse order
DROP INDEX IF EXISTS idx_intents_code;
DROP INDEX IF EXISTS idx_message_tags_tag_id;
DROP INDEX IF EXISTS idx_message_intents_intent_id;
DROP INDEX IF EXISTS idx_issues_created_at;
DROP INDEX IF EXISTS idx_issues_external_id;
DROP INDEX IF EXISTS idx_message_analysis_message_id;
DROP INDEX IF EXISTS idx_messages_published_at;
DROP INDEX IF EXISTS idx_messages_issue_id;

