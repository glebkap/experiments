import { format, formatDistanceToNow, parseISO } from 'date-fns';
import { ru } from 'date-fns/locale';

/**
 * Format date to readable format
 */
export const formatDate = (dateString: string | undefined, formatStr: string = 'dd.MM.yyyy HH:mm'): string => {
  if (!dateString) return '-';
  try {
    const date = typeof dateString === 'string' ? parseISO(dateString) : dateString;
    return format(date, formatStr, { locale: ru });
  } catch {
    return dateString;
  }
};

/**
 * Format date to relative time (e.g., "2 hours ago")
 */
export const formatRelativeTime = (dateString: string | undefined): string => {
  if (!dateString) return '-';
  try {
    const date = typeof dateString === 'string' ? parseISO(dateString) : dateString;
    return formatDistanceToNow(date, { addSuffix: true, locale: ru });
  } catch {
    return dateString;
  }
};

/**
 * Format confidence score to percentage
 */
export const formatConfidence = (confidence: number | undefined): string => {
  if (confidence === undefined || confidence === null) return '-';
  return `${Math.round(confidence * 100)}%`;
};

/**
 * Truncate text to max length
 */
export const truncateText = (text: string | undefined, maxLength: number = 100): string => {
  if (!text) return '-';
  if (text.length <= maxLength) return text;
  return `${text.substring(0, maxLength)}...`;
};

/**
 * Format file size
 */
export const formatFileSize = (bytes: number): string => {
  if (bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return `${parseFloat((bytes / Math.pow(k, i)).toFixed(2))} ${sizes[i]}`;
};

/**
 * Format duration in seconds to human readable
 */
export const formatDuration = (seconds: number | undefined): string => {
  if (!seconds) return '-';
  if (seconds < 60) return `${seconds} сек`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)} мин ${seconds % 60} сек`;
  const hours = Math.floor(seconds / 3600);
  const mins = Math.floor((seconds % 3600) / 60);
  return `${hours} ч ${mins} мин`;
};

/**
 * Format number with spaces as thousands separator
 */
export const formatNumber = (num: number | undefined): string => {
  if (num === undefined || num === null) return '-';
  return num.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ' ');
};

/**
 * Shorten UUID (show first 8 chars)
 */
export const shortenUUID = (uuid: string | undefined): string => {
  if (!uuid) return '-';
  return uuid.substring(0, 8);
};
