import React from 'react';
import { Chip, Tooltip } from '@mui/material';
import type { TagType } from '@/api/types';

interface TagBadgeProps {
  name: string;
  type?: TagType;
  onClick?: () => void;
  onDelete?: () => void;
  size?: 'small' | 'medium';
  tooltip?: string;
}

const TAG_COLORS: Record<TagType, 'default' | 'primary' | 'secondary' | 'success'> = {
  auto: 'primary',
  okdesk: 'secondary',
  manual: 'success',
};

const TagBadge: React.FC<TagBadgeProps> = ({
  name,
  type = 'auto',
  onClick,
  onDelete,
  size = 'small',
  tooltip,
}) => {
  const chip = (
    <Chip
      label={name}
      color={TAG_COLORS[type]}
      size={size}
      onClick={onClick}
      onDelete={onDelete}
      sx={{
        cursor: onClick ? 'pointer' : 'default',
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

export default TagBadge;
