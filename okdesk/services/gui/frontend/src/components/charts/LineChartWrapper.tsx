import React from 'react';
import { Box, Typography, Paper, useTheme } from '@mui/material';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import { CHART_COLORS } from '@/utils/constants';

export interface LineChartData {
  name: string;
  [key: string]: string | number;
}

export interface LineChartSeries {
  dataKey: string;
  name: string;
  color?: string;
  strokeWidth?: number;
}

interface LineChartWrapperProps {
  data: LineChartData[];
  series: LineChartSeries[];
  title?: string;
  height?: number;
  xAxisLabel?: string;
  yAxisLabel?: string;
  showLegend?: boolean;
  showGrid?: boolean;
  showDots?: boolean;
}

const LineChartWrapper: React.FC<LineChartWrapperProps> = ({
  data,
  series,
  title,
  height = 300,
  xAxisLabel,
  yAxisLabel,
  showLegend = true,
  showGrid = true,
  showDots = true,
}) => {
  const theme = useTheme();

  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      return (
        <Paper
          sx={{
            padding: 1.5,
            backgroundColor: 'background.paper',
            border: `1px solid ${theme.palette.divider}`,
          }}
        >
          <Typography variant="body2" fontWeight="medium" gutterBottom>
            {label}
          </Typography>
          {payload.map((entry: any, index: number) => (
            <Typography
              key={index}
              variant="body2"
              sx={{ color: entry.color }}
            >
              {entry.name}: {entry.value}
            </Typography>
          ))}
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
        <LineChart
          data={data}
          margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
        >
          {showGrid && <CartesianGrid strokeDasharray="3 3" />}
          <XAxis
            dataKey="name"
            label={
              xAxisLabel
                ? { value: xAxisLabel, position: 'insideBottom', offset: -5 }
                : undefined
            }
          />
          <YAxis
            label={
              yAxisLabel
                ? { value: yAxisLabel, angle: -90, position: 'insideLeft' }
                : undefined
            }
          />
          <Tooltip content={<CustomTooltip />} />
          {showLegend && <Legend />}
          {series.map((s, index) => (
            <Line
              key={s.dataKey}
              type="monotone"
              dataKey={s.dataKey}
              name={s.name}
              stroke={s.color || CHART_COLORS[index % CHART_COLORS.length]}
              strokeWidth={s.strokeWidth || 2}
              dot={showDots}
              activeDot={{ r: 6 }}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </Box>
  );
};

export default LineChartWrapper;
