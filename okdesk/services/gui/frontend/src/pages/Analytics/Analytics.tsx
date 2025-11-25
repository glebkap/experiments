import React, { useState, useMemo } from 'react';
import {
  Container,
  Box,
  Typography,
  Paper,
  Tabs,
  Tab,
  Card,
  CardContent,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Grid,
} from '@mui/material';
import PieChartWrapper, { type PieChartData } from '@/components/charts/PieChartWrapper';
import LineChartWrapper, { type LineChartData, type LineChartSeries } from '@/components/charts/LineChartWrapper';
import { mockSourceStats, mockTimelineStats } from '@/api/mocks';
import { formatNumber } from '@/utils/formatters';
import { SOURCE_TYPES, CHART_COLORS } from '@/utils/constants';

const Analytics: React.FC = () => {
  const [activeTab, setActiveTab] = useState(0);

  // Prepare data for source distribution pie chart
  const sourceDistributionData: PieChartData[] = useMemo(
    () =>
      mockSourceStats.map((stat) => ({
        name: stat.source_name,
        value: stat.issues_count,
      })),
    []
  );

  // Prepare data for timeline chart
  const timelineChartData: LineChartData[] = useMemo(
    () =>
      mockTimelineStats.map((stat) => ({
        name: stat.date,
        issues: stat.issues_count,
        messages: stat.messages_count,
      })),
    []
  );

  const timelineChartSeries: LineChartSeries[] = [
    {
      dataKey: 'issues',
      name: 'Обращения',
      color: CHART_COLORS[0],
    },
    {
      dataKey: 'messages',
      name: 'Сообщения',
      color: CHART_COLORS[1],
    },
  ];

  // Calculate totals
  const totalIssues = mockSourceStats.reduce((sum, stat) => sum + stat.issues_count, 0);
  const totalMessages = mockSourceStats.reduce((sum, stat) => sum + stat.messages_count, 0);
  const avgMessagesPerIssue = totalIssues > 0 ? totalMessages / totalIssues : 0;

  return (
    <Container maxWidth="xl">
      <Box sx={{ py: 4 }}>
        <Typography variant="h4" gutterBottom>
          Аналитика
        </Typography>
        <Typography variant="body1" color="text.secondary" paragraph>
          Статистика и визуализация данных по обращениям
        </Typography>

        {/* Summary Cards */}
        <Grid container spacing={3} sx={{ mb: 4 }}>
          <Grid size={{ xs: 12, sm: 6, md: 4 }}>
            <Card>
              <CardContent>
                <Typography color="text.secondary" gutterBottom>
                  Всего обращений
                </Typography>
                <Typography variant="h4">{formatNumber(totalIssues)}</Typography>
              </CardContent>
            </Card>
          </Grid>

          <Grid size={{ xs: 12, sm: 6, md: 4 }}>
            <Card>
              <CardContent>
                <Typography color="text.secondary" gutterBottom>
                  Всего сообщений
                </Typography>
                <Typography variant="h4">{formatNumber(totalMessages)}</Typography>
              </CardContent>
            </Card>
          </Grid>

          <Grid size={{ xs: 12, sm: 6, md: 4 }}>
            <Card>
              <CardContent>
                <Typography color="text.secondary" gutterBottom>
                  Среднее сообщений на обращение
                </Typography>
                <Typography variant="h4">{avgMessagesPerIssue.toFixed(1)}</Typography>
              </CardContent>
            </Card>
          </Grid>
        </Grid>

        {/* Tabs */}
        <Paper sx={{ mt: 3 }}>
          <Tabs
            value={activeTab}
            onChange={(_, newValue) => setActiveTab(newValue)}
            sx={{ borderBottom: 1, borderColor: 'divider' }}
          >
            <Tab label="По источникам" />
            <Tab label="Динамика по времени" />
          </Tabs>

          {/* Tab 1: Sources */}
          {activeTab === 0 && (
            <Box sx={{ p: 3 }}>
              <Grid container spacing={3}>
                {/* Pie Chart */}
                <Grid size={{ xs: 12, md: 6 }}>
                  <Paper sx={{ p: 2 }}>
                    <PieChartWrapper
                      title="Распределение обращений по источникам"
                      data={sourceDistributionData}
                      height={350}
                    />
                  </Paper>
                </Grid>

                {/* Stats Table */}
                <Grid size={{ xs: 12, md: 6 }}>
                  <Paper sx={{ p: 2 }}>
                    <Typography variant="h6" gutterBottom>
                      Детальная статистика
                    </Typography>
                    <TableContainer>
                      <Table size="small">
                        <TableHead>
                          <TableRow>
                            <TableCell>Источник</TableCell>
                            <TableCell align="right">Обращений</TableCell>
                            <TableCell align="right">Сообщений</TableCell>
                            <TableCell align="right">Среднее</TableCell>
                          </TableRow>
                        </TableHead>
                        <TableBody>
                          {mockSourceStats.map((stat) => (
                            <TableRow key={stat.source_id}>
                              <TableCell>
                                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                  <Box
                                    sx={{
                                      width: 12,
                                      height: 12,
                                      borderRadius: '50%',
                                      bgcolor: SOURCE_TYPES[stat.source_type].color,
                                    }}
                                  />
                                  {stat.source_name}
                                </Box>
                              </TableCell>
                              <TableCell align="right">
                                {formatNumber(stat.issues_count)}
                              </TableCell>
                              <TableCell align="right">
                                {formatNumber(stat.messages_count)}
                              </TableCell>
                              <TableCell align="right">
                                {stat.issues_count > 0
                                  ? (stat.messages_count / stat.issues_count).toFixed(1)
                                  : '-'}
                              </TableCell>
                            </TableRow>
                          ))}
                          <TableRow sx={{ fontWeight: 'bold' }}>
                            <TableCell sx={{ fontWeight: 'bold' }}>Итого</TableCell>
                            <TableCell align="right" sx={{ fontWeight: 'bold' }}>
                              {formatNumber(totalIssues)}
                            </TableCell>
                            <TableCell align="right" sx={{ fontWeight: 'bold' }}>
                              {formatNumber(totalMessages)}
                            </TableCell>
                            <TableCell align="right" sx={{ fontWeight: 'bold' }}>
                              {avgMessagesPerIssue.toFixed(1)}
                            </TableCell>
                          </TableRow>
                        </TableBody>
                      </Table>
                    </TableContainer>
                  </Paper>
                </Grid>
              </Grid>
            </Box>
          )}

          {/* Tab 2: Timeline */}
          {activeTab === 1 && (
            <Box sx={{ p: 3 }}>
              <Paper sx={{ p: 2 }}>
                <LineChartWrapper
                  title="Динамика обращений и сообщений"
                  data={timelineChartData}
                  series={timelineChartSeries}
                  height={400}
                  xAxisLabel="Дата"
                  yAxisLabel="Количество"
                />
              </Paper>

              {/* Timeline Stats Table */}
              <Paper sx={{ p: 2, mt: 3 }}>
                <Typography variant="h6" gutterBottom>
                  Данные по датам
                </Typography>
                <TableContainer>
                  <Table size="small">
                    <TableHead>
                      <TableRow>
                        <TableCell>Дата</TableCell>
                        <TableCell align="right">Обращений</TableCell>
                        <TableCell align="right">Сообщений</TableCell>
                        <TableCell align="right">Среднее сообщений</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {mockTimelineStats.map((stat) => (
                        <TableRow key={stat.date}>
                          <TableCell>{stat.date}</TableCell>
                          <TableCell align="right">
                            {formatNumber(stat.issues_count)}
                          </TableCell>
                          <TableCell align="right">
                            {formatNumber(stat.messages_count)}
                          </TableCell>
                          <TableCell align="right">
                            {stat.issues_count > 0
                              ? (stat.messages_count / stat.issues_count).toFixed(1)
                              : '-'}
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>
              </Paper>
            </Box>
          )}
        </Paper>
      </Box>
    </Container>
  );
};

export default Analytics;
