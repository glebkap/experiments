import React, { useState, useEffect } from 'react';
import {
  Box,
  Container,
  Card,
  CardContent,
  Typography,
  Paper,
} from '@mui/material';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
import AssignmentIcon from '@mui/icons-material/Assignment';
import MessageIcon from '@mui/icons-material/Message';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import { mockProcessingStats, mockImports } from '@/api/mocks';
import type { ProcessingStats, Import } from '@/api/types';
import { formatNumber, formatDate, formatDuration } from '@/utils/formatters';
import LoadingSpinner from '@/components/common/LoadingSpinner';
import ErrorMessage from '@/components/common/ErrorMessage';
import StatusBadge from '@/components/common/StatusBadge';
import DataTable from '@/components/tables/DataTable';
import type { Column } from '@/components/tables/DataTable';

interface StatCardProps {
  title: string;
  value: string | number;
  icon: React.ReactNode;
  color: string;
}

const StatCard: React.FC<StatCardProps> = ({ title, value, icon, color }) => {
  return (
    <Card elevation={2}>
      <CardContent>
        <Box display="flex" alignItems="center" justifyContent="space-between">
          <Box>
            <Typography variant="body2" color="text.secondary" gutterBottom>
              {title}
            </Typography>
            <Typography variant="h4" fontWeight="bold">
              {value}
            </Typography>
          </Box>
          <Box
            sx={{
              backgroundColor: color,
              borderRadius: 2,
              p: 1.5,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
            }}
          >
            {icon}
          </Box>
        </Box>
      </CardContent>
    </Card>
  );
};

const Dashboard: React.FC = () => {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [stats, setStats] = useState<ProcessingStats | null>(null);
  const [recentImports, setRecentImports] = useState<Import[]>([]);

  useEffect(() => {
    // Simulate API call
    const fetchData = async () => {
      try {
        setLoading(true);
        // In real app, this would be an API call
        await new Promise((resolve) => setTimeout(resolve, 500));
        setStats(mockProcessingStats);
        setRecentImports(mockImports.slice(0, 5));
        setError(null);
      } catch (err) {
        setError('Не удалось загрузить данные');
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  if (loading) {
    return <LoadingSpinner message="Загрузка статистики..." />;
  }

  if (error) {
    return <ErrorMessage message={error} />;
  }

  if (!stats) {
    return <ErrorMessage message="Нет данных для отображения" />;
  }

  const columns: Column<Import>[] = [
    {
      id: 'filename',
      label: 'Файл',
      sortable: true,
      render: (row) => row.filename || '-',
    },
    {
      id: 'status',
      label: 'Статус',
      sortable: true,
      align: 'center',
      render: (row) => <StatusBadge status={row.status} />,
    },
    {
      id: 'started_at',
      label: 'Начало',
      sortable: true,
      render: (row) => formatDate(row.started_at),
    },
    {
      id: 'duration',
      label: 'Длительность',
      sortable: false,
      render: (row) =>
        row.stats?.duration_seconds
          ? formatDuration(row.stats.duration_seconds)
          : '-',
    },
    {
      id: 'stats',
      label: 'Обработано',
      sortable: false,
      align: 'right',
      render: (row) => {
        if (!row.stats) return '-';
        return (
          <Box>
            <Typography variant="body2">
              Тикеты: {formatNumber(row.stats.new_issues || 0)} /{' '}
              {formatNumber(row.stats.total_issues || 0)}
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Сообщения: {formatNumber(row.stats.new_messages || 0)} /{' '}
              {formatNumber(row.stats.total_messages || 0)}
            </Typography>
          </Box>
        );
      },
    },
  ];

  return (
    <Container maxWidth="xl" sx={{ py: 3 }}>
      <Typography variant="h4" gutterBottom fontWeight="bold">
        Панель управления
      </Typography>
      <Typography variant="body1" color="text.secondary" paragraph>
        Обзор основных метрик и последних импортов
      </Typography>

      {/* Stats Cards */}
      <Box
        sx={{
          display: 'grid',
          gridTemplateColumns: {
            xs: '1fr',
            sm: '1fr 1fr',
            md: '1fr 1fr 1fr 1fr',
          },
          gap: 3,
          mb: 4,
        }}
      >
        <StatCard
          title="Всего тикетов"
          value={formatNumber(stats.total_issues)}
          icon={<AssignmentIcon sx={{ color: '#fff', fontSize: 32 }} />}
          color="rgba(25, 118, 210, 0.2)"
        />
        <StatCard
          title="Обработано тикетов"
          value={formatNumber(stats.processed_issues)}
          icon={<CheckCircleIcon sx={{ color: '#fff', fontSize: 32 }} />}
          color="rgba(76, 175, 80, 0.2)"
        />
        <StatCard
          title="Всего сообщений"
          value={formatNumber(stats.total_messages)}
          icon={<MessageIcon sx={{ color: '#fff', fontSize: 32 }} />}
          color="rgba(255, 152, 0, 0.2)"
        />
        <StatCard
          title="Процент обработки"
          value={`${stats.processing_rate || 0}%`}
          icon={<TrendingUpIcon sx={{ color: '#fff', fontSize: 32 }} />}
          color="rgba(156, 39, 176, 0.2)"
        />
      </Box>

      {/* Processing Progress */}
      <Paper sx={{ p: 3, mb: 4 }}>
        <Typography variant="h6" gutterBottom>
          Прогресс обработки
        </Typography>
        <Box
          sx={{
            display: 'grid',
            gridTemplateColumns: { xs: '1fr', md: '1fr 1fr' },
            gap: 3,
          }}
        >
          <Box>
            <Typography variant="body2" color="text.secondary" gutterBottom>
              Тикеты
            </Typography>
            <Box display="flex" alignItems="baseline" gap={1}>
              <Typography variant="h5" fontWeight="bold">
                {formatNumber(stats.processed_issues)}
              </Typography>
              <Typography variant="body1" color="text.secondary">
                / {formatNumber(stats.total_issues)}
              </Typography>
            </Box>
            <Typography variant="body2" color="text.secondary">
              Необработано: {formatNumber(stats.unprocessed_issues)}
            </Typography>
          </Box>
          <Box>
            <Typography variant="body2" color="text.secondary" gutterBottom>
              Сообщения
            </Typography>
            <Box display="flex" alignItems="baseline" gap={1}>
              <Typography variant="h5" fontWeight="bold">
                {formatNumber(stats.analyzed_messages)}
              </Typography>
              <Typography variant="body1" color="text.secondary">
                / {formatNumber(stats.total_messages)}
              </Typography>
            </Box>
            <Typography variant="body2" color="text.secondary">
              Не проанализировано:{' '}
              {formatNumber(stats.total_messages - stats.analyzed_messages)}
            </Typography>
          </Box>
        </Box>
      </Paper>

      {/* Recent Imports */}
      <Paper sx={{ p: 3 }}>
        <Typography variant="h6" gutterBottom>
          Последние импорты
        </Typography>
        <DataTable
          columns={columns}
          rows={recentImports}
          getRowId={(row) => row.id}
          emptyMessage="Нет импортов для отображения"
        />
      </Paper>
    </Container>
  );
};

export default Dashboard;
