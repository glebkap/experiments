import type {
  Issue,
  MessageWithAnalysis,
  Import,
  Cluster,
  ProcessingStats,
  SourceStats,
  TimelineStats,
} from './types';

// Mock data for development (until API Gateway is ready)

export const mockIssues: Issue[] = [
  {
    id: '1',
    external_id: 'OKDESK-1001',
    source_id: 'source-1',
    title: 'Проблема с оплатой',
    description: 'Не могу оплатить заказ через Сбербанк',
    status: 'opened',
    priority: 1,
    created_at: '2025-11-20T10:30:00Z',
    updated_at: '2025-11-20T10:30:00Z',
  },
  {
    id: '2',
    external_id: 'OKDESK-1002',
    source_id: 'source-1',
    title: 'Не работает личный кабинет',
    description: 'Не могу войти в личный кабинет, пишет ошибку авторизации',
    status: 'wait',
    priority: 2,
    created_at: '2025-11-19T14:20:00Z',
    updated_at: '2025-11-19T14:20:00Z',
  },
  {
    id: '3',
    external_id: 'OKDESK-1003',
    source_id: 'source-1',
    title: 'Запрос на добавление функции',
    description: 'Хотелось бы видеть историю заказов за весь период',
    status: 'completed',
    priority: 3,
    created_at: '2025-11-18T09:15:00Z',
    updated_at: '2025-11-18T16:45:00Z',
    completed_at: '2025-11-18T16:45:00Z',
  },
];

export const mockMessages: MessageWithAnalysis[] = [
  {
    id: 'msg-1',
    issue_id: '1',
    external_id: 'comment-1',
    author_name: 'Иван Иванов',
    author_type: 'contact',
    content: 'Здравствуйте! Пытаюсь оплатить заказ через Сбербанк, но постоянно выдает ошибку.',
    is_public: true,
    published_at: '2025-11-20T10:30:00Z',
    analysis: {
      id: 'analysis-1',
      message_id: 'msg-1',
      analyzed_at: '2025-11-20T10:35:00Z',
      reasoning: 'Пользователь сообщает о технической проблеме с оплатой через конкретный банк',
      intents: [
        {
          id: 'mi-1',
          message_analysis_id: 'analysis-1',
          intent_id: 'intent-1',
          confidence: 0.95,
          intent: {
            id: 'intent-1',
            code: 'payment_issue',
            name: 'Проблема с оплатой',
            description: 'Пользователь сообщает о проблемах при оплате',
            created_at: '2025-11-01T00:00:00Z',
          },
        },
      ],
      tags: [
        {
          id: 'mt-1',
          message_analysis_id: 'analysis-1',
          tag_id: 'tag-1',
          confidence: 0.9,
          tag: {
            id: 'tag-1',
            name: 'сбербанк',
            type: 'auto',
            created_at: '2025-11-20T10:35:00Z',
          },
        },
        {
          id: 'mt-2',
          message_analysis_id: 'analysis-1',
          tag_id: 'tag-2',
          confidence: 0.85,
          tag: {
            id: 'tag-2',
            name: 'оплата',
            type: 'auto',
            created_at: '2025-11-20T10:35:00Z',
          },
        },
      ],
    },
  },
  {
    id: 'msg-2',
    issue_id: '1',
    external_id: 'comment-2',
    author_name: 'Поддержка',
    author_type: 'employee',
    content: 'Добрый день! Подскажите, какую ошибку вы видите? Можете прислать скриншот?',
    is_public: true,
    published_at: '2025-11-20T10:40:00Z',
  },
];

export const mockImports: Import[] = [
  {
    id: 'import-1',
    source_id: 'source-1',
    filename: 'okdesk_export_2025_11_20.jsonl',
    started_at: '2025-11-20T08:00:00Z',
    completed_at: '2025-11-20T08:15:00Z',
    status: 'completed',
    stats: {
      total_issues: 150,
      new_issues: 100,
      updated_issues: 50,
      total_messages: 500,
      new_messages: 300,
      duration_seconds: 900,
    },
  },
  {
    id: 'import-2',
    source_id: 'source-2',
    filename: 'telegram_export.json',
    started_at: '2025-11-19T14:00:00Z',
    completed_at: '2025-11-19T14:05:00Z',
    status: 'completed',
    stats: {
      total_issues: 50,
      new_issues: 50,
      total_messages: 200,
      new_messages: 200,
      duration_seconds: 300,
    },
  },
  {
    id: 'import-3',
    source_id: 'source-1',
    filename: 'okdesk_failed.jsonl',
    started_at: '2025-11-18T10:00:00Z',
    status: 'failed',
    error_message: 'Invalid JSON format at line 45',
  },
];

export const mockClusters: Cluster[] = [
  {
    id: 'cluster-1',
    cluster_label: 0,
    name: 'Проблемы с оплатой',
    description: 'Группа обращений связанных с оплатой и платежами',
    size: 45,
    created_at: '2025-11-15T12:00:00Z',
    updated_at: '2025-11-15T12:00:00Z',
  },
  {
    id: 'cluster-2',
    cluster_label: 1,
    name: 'Авторизация и доступ',
    description: 'Проблемы со входом в систему, восстановление пароля',
    size: 32,
    created_at: '2025-11-15T12:00:00Z',
    updated_at: '2025-11-15T12:00:00Z',
  },
  {
    id: 'cluster-3',
    cluster_label: 2,
    name: 'Запросы функций',
    description: 'Предложения по улучшению и новым функциям',
    size: 28,
    created_at: '2025-11-15T12:00:00Z',
    updated_at: '2025-11-15T12:00:00Z',
  },
];

export const mockProcessingStats: ProcessingStats = {
  total_issues: 250,
  processed_issues: 200,
  unprocessed_issues: 50,
  total_messages: 1000,
  analyzed_messages: 800,
  processing_rate: 80,
};

export const mockSourceStats: SourceStats[] = [
  {
    source_id: 'source-1',
    source_name: 'OKDesk Production',
    source_type: 'okdesk',
    issues_count: 200,
    messages_count: 800,
  },
  {
    source_id: 'source-2',
    source_name: 'Telegram Support',
    source_type: 'telegram',
    issues_count: 50,
    messages_count: 200,
  },
];

export const mockTimelineStats: TimelineStats[] = [
  { date: '2025-11-15', issues_count: 25, messages_count: 100 },
  { date: '2025-11-16', issues_count: 30, messages_count: 120 },
  { date: '2025-11-17', issues_count: 22, messages_count: 88 },
  { date: '2025-11-18', issues_count: 28, messages_count: 112 },
  { date: '2025-11-19', issues_count: 35, messages_count: 140 },
  { date: '2025-11-20', issues_count: 40, messages_count: 160 },
  { date: '2025-11-21', issues_count: 32, messages_count: 128 },
];

// Mock API flag - set to false when API Gateway is ready
export const USE_MOCK_DATA = true;
