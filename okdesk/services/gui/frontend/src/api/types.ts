// Enums
export type SourceType = 'okdesk' | 'telegram';
export type IssueStatus = 'opened' | 'wait' | 'completed' | 'closed';
export type AuthorType = 'employee' | 'contact' | 'user';
export type ImportStatus = 'in_progress' | 'completed' | 'failed';
export type TagType = 'auto' | 'okdesk' | 'manual';

// Source
export interface Source {
  id: string;
  name: string;
  type: SourceType;
  config?: Record<string, unknown>;
  created_at: string;
}

// Issue
export interface Issue {
  id: string;
  external_id: string;
  source_id: string;
  title?: string;
  description?: string;
  status?: IssueStatus;
  priority?: number;
  created_at?: string;
  updated_at?: string;
  completed_at?: string;
  source?: Source;
}

// Message
export interface Message {
  id: string;
  issue_id: string;
  external_id: string;
  author_id?: string;
  author_name?: string;
  author_type?: AuthorType;
  content: string;
  is_public: boolean;
  published_at?: string;
}

// Intent
export interface Intent {
  id: string;
  code: string;
  name: string;
  description: string;
  created_at: string;
}

// Tag
export interface Tag {
  id: string;
  name: string;
  type?: TagType;
  source?: string;
  created_at: string;
}

// Message Analysis
export interface MessageAnalysis {
  id: string;
  message_id: string;
  analyzed_at: string;
  reasoning?: string;
  intents: MessageIntent[];
  tags: MessageTag[];
}

export interface MessageIntent {
  id: string;
  message_analysis_id: string;
  intent_id: string;
  confidence: number;
  intent?: Intent;
}

export interface MessageTag {
  id: string;
  message_analysis_id: string;
  tag_id: string;
  confidence: number;
  tag?: Tag;
}

// Cluster
export interface Cluster {
  id: string;
  cluster_label: number;
  name?: string;
  description?: string;
  size: number;
  created_at: string;
  updated_at: string;
}

// Import
export interface Import {
  id: string;
  source_id: string;
  filename?: string;
  file_path?: string;
  started_at: string;
  completed_at?: string;
  status: ImportStatus;
  stats?: ImportStats;
  error_message?: string;
  source?: Source;
}

export interface ImportStats {
  total_issues?: number;
  new_issues?: number;
  updated_issues?: number;
  total_messages?: number;
  new_messages?: number;
  duration_seconds?: number;
}

// API Request/Response types

// Import endpoints
export interface ImportRequest {
  file: File;
  source_type: SourceType;
}

export interface ImportResponse {
  import_id: string;
  message: string;
}

export interface ImportListResponse {
  imports: Import[];
  total: number;
}

// Issues endpoints
export interface IssuesListParams {
  status?: IssueStatus;
  source_id?: string;
  limit?: number;
  offset?: number;
  search?: string;
  start_date?: string;
  end_date?: string;
}

export interface IssuesListResponse {
  issues: Issue[];
  total: number;
}

export interface IssueDetailResponse {
  issue: Issue;
  messages: MessageWithAnalysis[];
}

export interface MessageWithAnalysis extends Message {
  analysis?: MessageAnalysis;
}

// Search endpoints
export interface SearchRequest {
  query: string;
  top_k?: number;
  min_similarity?: number;
  filters?: SearchFilters;
}

export interface SearchFilters {
  status?: IssueStatus[];
  source_ids?: string[];
  intent_ids?: string[];
  tag_ids?: string[];
  start_date?: string;
  end_date?: string;
}

export interface SearchResult {
  issue_id: string;
  similarity: number;
  text_snippet: string;
  issue?: Issue;
}

export interface SearchResponse {
  results: SearchResult[];
  total: number;
}

// Stats endpoints
export interface ProcessingStats {
  total_issues: number;
  processed_issues: number;
  unprocessed_issues: number;
  total_messages: number;
  analyzed_messages: number;
  processing_rate?: number;
}

export interface ClusterStats {
  total_clusters: number;
  clusters: ClusterInfo[];
}

export interface ClusterInfo {
  cluster_id: string;
  cluster_label: number;
  name?: string;
  size: number;
  avg_distance: number;
  top_tags: string[];
}

export interface SourceStats {
  source_id: string;
  source_name: string;
  source_type: SourceType;
  issues_count: number;
  messages_count: number;
}

export interface TimelineStats {
  date: string;
  issues_count: number;
  messages_count: number;
}

// Export endpoints
export interface ExportRequest {
  format: 'csv' | 'json';
  filters?: SearchFilters;
  fields?: string[];
}

export interface ExportResponse {
  file_url: string;
  filename: string;
  size_bytes: number;
}

// Pipeline endpoints
export interface PipelineProcessRequest {
  batch_size?: number;
  device?: string;
}

export interface PipelineProcessResponse {
  processed_count: number;
  duration: number;
  stats: {
    stage_durations: Record<string, number>;
  };
}

export interface PipelineStatusResponse {
  is_running: boolean;
  current_batch?: number;
  total_batches?: number;
  progress_percent?: number;
}

// Clustering endpoints
export interface ClusteringRunRequest {
  method: 'hdbscan' | 'kmeans';
  min_cluster_size?: number;
  n_clusters?: number;
}

export interface ClusteringRunResponse {
  clusters_created: number;
  duration: number;
  silhouette_score?: number;
}

// Health check
export interface HealthResponse {
  status: 'healthy' | 'unhealthy';
  services: Record<string, ServiceHealth>;
}

export interface ServiceHealth {
  status: 'up' | 'down';
  response_time_ms?: number;
  error?: string;
}
