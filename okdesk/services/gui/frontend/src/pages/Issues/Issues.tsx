import React, { useState, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Container,
  Box,
  Typography,
  Paper,
  TextField,
  MenuItem,
  Select,
  FormControl,
  InputLabel,
  Chip,
  InputAdornment,
} from '@mui/material';
import SearchIcon from '@mui/icons-material/Search';
import DataTable, { type Column } from '@/components/tables/DataTable';
import { mockIssues } from '@/api/mocks';
import type { Issue, IssueStatus, SourceType } from '@/api/types';
import { formatDate } from '@/utils/formatters';
import { STATUS_COLORS, PRIORITY_COLORS, SOURCE_TYPES } from '@/utils/constants';
import { useDebounce } from '@/hooks/useDebounce';

const Issues: React.FC = () => {
  const navigate = useNavigate();
  const [searchQuery, setSearchQuery] = useState('');
  const [statusFilter, setStatusFilter] = useState<IssueStatus | 'all'>('all');
  const debouncedSearch = useDebounce(searchQuery, 500);

  const getStatusLabel = (status: IssueStatus | undefined): string => {
    if (!status) return '-';
    const labels: Record<IssueStatus, string> = {
      opened: 'Открыто',
      wait: 'Ожидание',
      completed: 'Завершено',
      closed: 'Закрыто',
    };
    return labels[status];
  };

  const getPriorityLabel = (priority: number | undefined): string => {
    if (!priority) return '-';
    const labels: Record<number, string> = {
      1: 'Критический',
      2: 'Высокий',
      3: 'Нормальный',
      4: 'Низкий',
    };
    return labels[priority] || priority.toString();
  };

  const getSourceType = (sourceId: string | undefined): SourceType => {
    // Mock logic - in real app would look up from sources
    return sourceId === 'source-1' ? 'okdesk' : 'telegram';
  };

  // Filter and search issues
  const filteredIssues = useMemo(() => {
    let filtered = mockIssues;

    // Apply status filter
    if (statusFilter !== 'all') {
      filtered = filtered.filter((issue) => issue.status === statusFilter);
    }

    // Apply search filter
    if (debouncedSearch) {
      const searchLower = debouncedSearch.toLowerCase();
      filtered = filtered.filter(
        (issue) =>
          issue.title?.toLowerCase().includes(searchLower) ||
          issue.description?.toLowerCase().includes(searchLower) ||
          issue.external_id.toLowerCase().includes(searchLower)
      );
    }

    return filtered;
  }, [statusFilter, debouncedSearch]);

  const handleRowClick = (issue: Issue) => {
    navigate(`/issues/${issue.id}`);
  };

  const columns: Column<Issue>[] = [
    {
      id: 'external_id',
      label: 'ID',
      sortable: true,
      width: 150,
      render: (row) => (
        <Typography variant="body2" fontFamily="monospace">
          {row.external_id}
        </Typography>
      ),
    },
    {
      id: 'title',
      label: 'Заголовок',
      sortable: true,
      render: (row) => row.title || '-',
    },
    {
      id: 'status',
      label: 'Статус',
      sortable: true,
      width: 120,
      render: (row) => (
        <Chip
          label={getStatusLabel(row.status)}
          size="small"
          sx={{
            backgroundColor: row.status ? STATUS_COLORS[row.status] : 'grey',
            color: 'white',
          }}
        />
      ),
    },
    {
      id: 'priority',
      label: 'Приоритет',
      sortable: true,
      width: 120,
      render: (row) =>
        row.priority ? (
          <Chip
            label={getPriorityLabel(row.priority)}
            size="small"
            sx={{
              backgroundColor: PRIORITY_COLORS[row.priority as keyof typeof PRIORITY_COLORS],
              color: 'white',
            }}
          />
        ) : (
          '-'
        ),
    },
    {
      id: 'source',
      label: 'Источник',
      width: 150,
      render: (row) => {
        const sourceType = getSourceType(row.source_id);
        return (
          <Chip
            label={SOURCE_TYPES[sourceType].label}
            size="small"
            sx={{
              backgroundColor: SOURCE_TYPES[sourceType].color,
              color: 'white',
            }}
          />
        );
      },
    },
    {
      id: 'created_at',
      label: 'Создано',
      sortable: true,
      width: 180,
      render: (row) => formatDate(row.created_at, 'dd.MM.yyyy HH:mm'),
    },
  ];

  return (
    <Container maxWidth="xl">
      <Box sx={{ py: 4 }}>
        <Typography variant="h4" gutterBottom>
          Обращения
        </Typography>
        <Typography variant="body1" color="text.secondary" paragraph>
          Список всех обращений из поддержки
        </Typography>

        {/* Filters */}
        <Paper sx={{ p: 2, mb: 3 }}>
          <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap' }}>
            <TextField
              placeholder="Поиск по заголовку, описанию или ID..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              sx={{ flexGrow: 1, minWidth: 300 }}
              InputProps={{
                startAdornment: (
                  <InputAdornment position="start">
                    <SearchIcon />
                  </InputAdornment>
                ),
              }}
            />

            <FormControl sx={{ minWidth: 150 }}>
              <InputLabel>Статус</InputLabel>
              <Select
                value={statusFilter}
                label="Статус"
                onChange={(e) => setStatusFilter(e.target.value as IssueStatus | 'all')}
              >
                <MenuItem value="all">Все</MenuItem>
                <MenuItem value="opened">Открыто</MenuItem>
                <MenuItem value="wait">Ожидание</MenuItem>
                <MenuItem value="completed">Завершено</MenuItem>
                <MenuItem value="closed">Закрыто</MenuItem>
              </Select>
            </FormControl>
          </Box>

          {(debouncedSearch || statusFilter !== 'all') && (
            <Box sx={{ mt: 2 }}>
              <Typography variant="body2" color="text.secondary">
                Найдено обращений: {filteredIssues.length}
              </Typography>
            </Box>
          )}
        </Paper>

        {/* Table */}
        <Box
          sx={{
            '& table tbody tr': {
              cursor: 'pointer',
            },
          }}
          onClick={(e) => {
            const target = e.target as HTMLElement;
            const row = target.closest('tr');
            if (row) {
              const rowIndex = Array.from(row.parentElement?.children || []).indexOf(row);
              if (rowIndex >= 0 && filteredIssues[rowIndex]) {
                handleRowClick(filteredIssues[rowIndex]);
              }
            }
          }}
        >
          <DataTable
            columns={columns}
            rows={filteredIssues}
            getRowId={(row) => row.id}
            emptyMessage={
              debouncedSearch || statusFilter !== 'all'
                ? 'По вашему запросу ничего не найдено'
                : 'Нет обращений'
            }
          />
        </Box>
      </Box>
    </Container>
  );
};

export default Issues;
