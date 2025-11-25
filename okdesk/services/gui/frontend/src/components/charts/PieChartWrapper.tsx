import React from 'react';
import { Box, Paper, Typography, useTheme } from '@mui/material';
import {
  PieChart,
  Pie,
  Cell,
  ResponsiveContainer,
  Legend,
  Tooltip,
} from 'recharts';
import { CHART_COLORS } from '@/utils/constants';

export interface PieChartData {
  name: string;
  value: number;
  [key: string]: string | number;
}

interface PieChartWrapperProps {
  data: PieChartData[];
  title?: string;
  height?: number;
  colors?: string[];
  showLegend?: boolean;
  showTooltip?: boolean;
}

const PieChartWrapper: React.FC<PieChartWrapperProps> = ({
  data,
  title,
  height = 300,
  colors = CHART_COLORS,
  showLegend = true,
  showTooltip = true,
}) => {
  const theme = useTheme();

  const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      return (
        <Paper
          sx={{
            padding: 1.5,
            backgroundColor: 'background.paper',
            border: `1px solid ${theme.palette.divider}`,
          }}
        >
          <Typography variant="body2" fontWeight="medium">
            {payload[0].name}
          </Typography>
          <Typography variant="body2" color="primary">
            {payload[0].value}
          </Typography>
        </Paper>
      );
    }
    return null;
  };

  return (
    <Box>
      {title && (
        <Typography variant="h6" gutterBottom>
          {title}
        </Typography>
      )}
      <ResponsiveContainer width="100%" height={height}>
        <PieChart>
          <Pie
            data={data}
            cx="50%"
            cy="50%"
            labelLine={false}
            label={({ name, percent }) =>
              `${name}: ${((percent || 0) * 100).toFixed(0)}%`
            }
            outerRadius={80}
            fill="#8884d8"
            dataKey="value"
          >
            {data.map((_entry, index) => (
              <Cell
                key={`cell-${index}`}
                fill={colors[index % colors.length]}
              />
            ))}
          </Pie>
          {showTooltip && <Tooltip content={<CustomTooltip />} />}
          {showLegend && <Legend />}
        </PieChart>
      </ResponsiveContainer>
    </Box>
  );
};

export default PieChartWrapper;
