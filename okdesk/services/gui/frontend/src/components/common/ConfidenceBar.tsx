import React from 'react';
import { Box, LinearProgress, Typography, Tooltip } from '@mui/material';
import { formatConfidence } from '@/utils/formatters';

interface ConfidenceBarProps {
  confidence: number;
  showLabel?: boolean;
  height?: number;
  tooltip?: string;
}

const ConfidenceBar: React.FC<ConfidenceBarProps> = ({
  confidence,
  showLabel = true,
  height = 8,
  tooltip,
}) => {
  const percentage = Math.round(confidence * 100);

  const getColor = (value: number): 'error' | 'warning' | 'success' => {
    if (value < 50) return 'error';
    if (value < 75) return 'warning';
    return 'success';
  };

  const bar = (
    <Box display="flex" alignItems="center" gap={1} width="100%">
      <Box flex={1}>
        <LinearProgress
          variant="determinate"
          value={percentage}
          color={getColor(percentage)}
          sx={{ height, borderRadius: 1 }}
        />
      </Box>
      {showLabel && (
        <Typography
          variant="body2"
          color="text.secondary"
          sx={{ minWidth: 45, textAlign: 'right' }}
        >
          {formatConfidence(confidence)}
        </Typography>
      )}
    </Box>
  );

  if (tooltip) {
    return (
      <Tooltip title={tooltip} arrow>
        {bar}
      </Tooltip>
    );
  }

  return bar;
};

export default ConfidenceBar;
