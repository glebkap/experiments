// API Configuration
export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api/v1';

// Pagination
export const DEFAULT_PAGE_SIZE = 20;
export const PAGE_SIZE_OPTIONS = [10, 20, 50, 100];

// Chart Colors
export const CHART_COLORS = [
  '#1976d2', // primary blue
  '#9c27b0', // purple
  '#f44336', // red
  '#4caf50', // green
  '#ff9800', // orange
  '#00bcd4', // cyan
  '#e91e63', // pink
  '#3f51b5', // indigo
  '#8bc34a', // light green
  '#ff5722', // deep orange
  '#673ab7', // deep purple
  '#009688', // teal
];

// Status Colors
export const STATUS_COLORS = {
  opened: '#1976d2', // blue
  wait: '#ff9800', // orange
  completed: '#4caf50', // green
  closed: '#9e9e9e', // grey
};

// Priority Colors
export const PRIORITY_COLORS = {
  1: '#f44336', // red - highest
  2: '#ff9800', // orange - high
  3: '#2196f3', // blue - normal
  4: '#4caf50', // green - low
};

// Import Status Colors
export const IMPORT_STATUS_COLORS = {
  in_progress: '#2196f3', // blue
  completed: '#4caf50', // green
  failed: '#f44336', // red
};

// Source Type Icons & Colors
export const SOURCE_TYPES = {
  okdesk: {
    label: 'OKDesk',
    color: '#1976d2',
  },
  telegram: {
    label: 'Telegram',
    color: '#0088cc',
  },
};

// File Upload
export const MAX_FILE_SIZE = 100 * 1024 * 1024; // 100 MB
export const ACCEPTED_FILE_TYPES = {
  okdesk: ['.jsonl'],
  telegram: ['.json'],
};

// Debounce Delays
export const SEARCH_DEBOUNCE_MS = 500;
export const FILTER_DEBOUNCE_MS = 300;

// Toast/Notification Durations
export const TOAST_DURATION = 5000; // 5 seconds

// Application
export const APP_NAME = import.meta.env.VITE_APP_NAME || 'Support Intent Analyzer';
