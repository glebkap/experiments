import apiClient, { createFormData } from './client';
import type {
  ImportResponse,
  ImportListResponse,
  Import,
  IssuesListParams,
  IssuesListResponse,
  IssueDetailResponse,
  SearchRequest,
  SearchResponse,
  ProcessingStats,
  ClusterStats,
  SourceStats,
  TimelineStats,
  ExportRequest,
  ExportResponse,
  PipelineProcessRequest,
  PipelineProcessResponse,
  PipelineStatusResponse,
  ClusteringRunRequest,
  ClusteringRunResponse,
  HealthResponse,
  Cluster,
} from './types';

// ============================================
// IMPORT ENDPOINTS (Parser Service)
// ============================================

export const importAPI = {
  // Upload OKDesk file
  uploadOKDesk: async (file: File): Promise<ImportResponse> => {
    const formData = createFormData(file);
    const response = await apiClient.post<ImportResponse>('/import/okdesk', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
    return response.data;
  },

  // Upload Telegram file
  uploadTelegram: async (file: File): Promise<ImportResponse> => {
    const formData = createFormData(file);
    const response = await apiClient.post<ImportResponse>('/import/telegram', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
    return response.data;
  },

  // Get import status
  getImportStatus: async (importId: string): Promise<Import> => {
    const response = await apiClient.get<Import>(`/import/${importId}`);
    return response.data;
  },

  // Get imports history
  getImports: async (params?: { limit?: number; offset?: number }): Promise<ImportListResponse> => {
    const response = await apiClient.get<ImportListResponse>('/imports', { params });
    return response.data;
  },
};

// ============================================
// ANALYZER ENDPOINTS - Pipeline & Clustering
// ============================================

export const pipelineAPI = {
  // Start pipeline processing
  processPipeline: async (request: PipelineProcessRequest): Promise<PipelineProcessResponse> => {
    const response = await apiClient.post<PipelineProcessResponse>(
      '/analyzer/pipeline/process',
      request
    );
    return response.data;
  },

  // Get pipeline status
  getPipelineStatus: async (): Promise<PipelineStatusResponse> => {
    const response = await apiClient.get<PipelineStatusResponse>('/analyzer/pipeline/status');
    return response.data;
  },
};

export const clusteringAPI = {
  // Run clustering
  runClustering: async (request: ClusteringRunRequest): Promise<ClusteringRunResponse> => {
    const response = await apiClient.post<ClusteringRunResponse>(
      '/analyzer/clustering/run',
      request
    );
    return response.data;
  },

  // Get clustering info
  getClusteringInfo: async (): Promise<{ clusters: Cluster[] }> => {
    const response = await apiClient.get<{ clusters: Cluster[] }>('/analyzer/clustering/info');
    return response.data;
  },
};

// ============================================
// ANALYZER ENDPOINTS - Search
// ============================================

export const searchAPI = {
  // Semantic search
  searchSimilar: async (request: SearchRequest): Promise<SearchResponse> => {
    const response = await apiClient.post<SearchResponse>('/analyzer/search/similar', request);
    return response.data;
  },

  // Fulltext search
  searchFulltext: async (params: {
    q: string;
    limit?: number;
    offset?: number;
  }): Promise<SearchResponse> => {
    const response = await apiClient.get<SearchResponse>('/analyzer/search/fulltext', { params });
    return response.data;
  },
};

// ============================================
// ANALYZER ENDPOINTS - Issues & Data
// ============================================

export const issuesAPI = {
  // Get issues list
  getIssues: async (params: IssuesListParams): Promise<IssuesListResponse> => {
    const response = await apiClient.get<IssuesListResponse>('/analyzer/issues', { params });
    return response.data;
  },

  // Get issue detail
  getIssueDetail: async (issueId: string): Promise<IssueDetailResponse> => {
    const response = await apiClient.get<IssueDetailResponse>(`/analyzer/issues/${issueId}`);
    return response.data;
  },

  // Get cluster issues
  getClusterIssues: async (
    clusterId: string,
    params?: { limit?: number; offset?: number }
  ): Promise<IssuesListResponse> => {
    const response = await apiClient.get<IssuesListResponse>(
      `/analyzer/clusters/${clusterId}/issues`,
      { params }
    );
    return response.data;
  },
};

// ============================================
// ANALYZER ENDPOINTS - Statistics
// ============================================

export const statsAPI = {
  // Get processing stats
  getProcessingStats: async (): Promise<ProcessingStats> => {
    const response = await apiClient.get<ProcessingStats>('/analyzer/stats/processing');
    return response.data;
  },

  // Get cluster stats
  getClusterStats: async (): Promise<ClusterStats> => {
    const response = await apiClient.get<ClusterStats>('/analyzer/stats/clusters');
    return response.data;
  },

  // Get source stats
  getSourceStats: async (): Promise<{ sources: SourceStats[] }> => {
    const response = await apiClient.get<{ sources: SourceStats[] }>('/analyzer/stats/sources');
    return response.data;
  },

  // Get timeline stats
  getTimelineStats: async (params?: {
    start_date?: string;
    end_date?: string;
  }): Promise<{ timeline: TimelineStats[] }> => {
    const response = await apiClient.get<{ timeline: TimelineStats[] }>(
      '/analyzer/stats/timeline',
      { params }
    );
    return response.data;
  },
};

// ============================================
// ANALYZER ENDPOINTS - Export
// ============================================

export const exportAPI = {
  // Export data
  exportData: async (request: ExportRequest): Promise<ExportResponse> => {
    const response = await apiClient.post<ExportResponse>('/analyzer/export', request, {
      responseType: 'blob',
    });

    // Create download link
    const blob = new Blob([response.data as unknown as BlobPart], {
      type: request.format === 'csv' ? 'text/csv' : 'application/json',
    });
    const url = window.URL.createObjectURL(blob);
    const filename = `export_${new Date().toISOString().split('T')[0]}.${request.format}`;

    return {
      file_url: url,
      filename,
      size_bytes: blob.size,
    };
  },
};

// ============================================
// HEALTH CHECK
// ============================================

export const healthAPI = {
  // Check overall health
  checkHealth: async (): Promise<HealthResponse> => {
    const response = await apiClient.get<HealthResponse>('/health');
    return response.data;
  },

  // Check parser service health
  checkParserHealth: async (): Promise<{ status: string }> => {
    const response = await apiClient.get<{ status: string }>('/health/parser');
    return response.data;
  },

  // Check analyzer service health
  checkAnalyzerHealth: async (): Promise<{ status: string }> => {
    const response = await apiClient.get<{ status: string }>('/health/analyzer');
    return response.data;
  },
};

// Export all APIs
export default {
  import: importAPI,
  pipeline: pipelineAPI,
  clustering: clusteringAPI,
  search: searchAPI,
  issues: issuesAPI,
  stats: statsAPI,
  export: exportAPI,
  health: healthAPI,
};
