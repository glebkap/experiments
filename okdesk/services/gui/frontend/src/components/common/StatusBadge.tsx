import React from 'react';
import { Chip, Tooltip } from '@mui/material';
import type { IssueStatus, ImportStatus } from '@/api/types';
import { STATUS_COLORS, IMPORT_STATUS_COLORS } from '@/utils/constants';

type Status = IssueStatus | ImportStatus;

interface StatusBadgeProps {
  status: Status;
  size?: 'small' | 'medium';
  tooltip?: string;
}

const STATUS_LABELS: Record<IssueStatus, string> = {
  opened: 'Открыто',
  wait: 'Ожидание',
  completed: 'Завершено',
  closed: 'Закрыто',
};

const IMPORT_STATUS_LABELS: Record<ImportStatus, string> = {
  in_progress: 'В процессе',
  completed: 'Завершено',
  failed: 'Ошибка',
};

const isIssueStatus = (status: Status): status is IssueStatus => {
  return ['opened', 'wait', 'completed', 'closed'].includes(status);
};

const StatusBadge: React.FC<StatusBadgeProps> = ({
  status,
  size = 'small',
  tooltip,
}) => {
  const label = isIssueStatus(status)
    ? STATUS_LABELS[status]
    : IMPORT_STATUS_LABELS[status as ImportStatus];

  const color = isIssueStatus(status)
    ? STATUS_COLORS[status]
    : IMPORT_STATUS_COLORS[status as ImportStatus];

  const chip = (
    <Chip
      label={label}
      size={size}
      sx={{
        backgroundColor: color,
        color: '#fff',
        fontWeight: 500,
        '& .MuiChip-label': {
          px: 1,
        },
      }}
    />
  );

  if (tooltip) {
    return (
      <Tooltip title={tooltip} arrow>
        {chip}
      </Tooltip>
    );
  }

  return chip;
};

export default StatusBadge;
