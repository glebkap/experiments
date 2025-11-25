import React, { useState, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Container,
  Box,
  Typography,
  Paper,
  TextField,
  Button,
  Collapse,
  Card,
  CardContent,
  CardActionArea,
  Chip,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  InputAdornment,
  Divider,
  Grid,
} from '@mui/material';
import SearchIcon from '@mui/icons-material/Search';
import FilterListIcon from '@mui/icons-material/FilterList';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import ExpandLessIcon from '@mui/icons-material/ExpandLess';
import { mockIssues } from '@/api/mocks';
import type { Issue, IssueStatus } from '@/api/types';
import { formatDate, truncateText } from '@/utils/formatters';
import { STATUS_COLORS } from '@/utils/constants';
import { useDebounce } from '@/hooks/useDebounce';

const Search: React.FC = () => {
  const navigate = useNavigate();
  const [searchQuery, setSearchQuery] = useState('');
  const [filtersExpanded, setFiltersExpanded] = useState(false);
  const [statusFilter, setStatusFilter] = useState<IssueStatus | 'all'>('all');
  const [dateFrom, setDateFrom] = useState('');
  const [dateTo, setDateTo] = useState('');

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

  // Filter results based on search and filters
  const searchResults = useMemo(() => {
    let filtered = mockIssues;

    // Apply search query
    if (debouncedSearch) {
      const searchLower = debouncedSearch.toLowerCase();
      filtered = filtered.filter(
        (issue) =>
          issue.title?.toLowerCase().includes(searchLower) ||
          issue.description?.toLowerCase().includes(searchLower) ||
          issue.external_id.toLowerCase().includes(searchLower)
      );
    }

    // Apply status filter
    if (statusFilter !== 'all') {
      filtered = filtered.filter((issue) => issue.status === statusFilter);
    }

    // Apply date filters
    if (dateFrom) {
      filtered = filtered.filter(
        (issue) => !issue.created_at || new Date(issue.created_at) >= new Date(dateFrom)
      );
    }

    if (dateTo) {
      filtered = filtered.filter(
        (issue) => !issue.created_at || new Date(issue.created_at) <= new Date(dateTo)
      );
    }

    return filtered;
  }, [debouncedSearch, statusFilter, dateFrom, dateTo]);

  const handleSearch = () => {
    // In real app, would trigger API search here
    // For now, results are filtered reactively via useMemo
  };

  const handleClearFilters = () => {
    setSearchQuery('');
    setStatusFilter('all');
    setDateFrom('');
    setDateTo('');
  };

  const handleIssueClick = (issue: Issue) => {
    navigate(`/issues/${issue.id}`);
  };

  const hasActiveFilters = statusFilter !== 'all' || dateFrom || dateTo;

  return (
    <Container maxWidth="xl">
      <Box sx={{ py: 4 }}>
        <Typography variant="h4" gutterBottom>
          Поиск обращений
        </Typography>
        <Typography variant="body1" color="text.secondary" paragraph>
          Найдите обращения по ключевым словам в заголовке и описании
        </Typography>

        {/* Search Bar */}
        <Paper sx={{ p: 3, mb: 3 }}>
          <Box sx={{ display: 'flex', gap: 2, mb: 2 }}>
            <TextField
              fullWidth
              placeholder="Введите поисковый запрос..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              onKeyPress={(e) => {
                if (e.key === 'Enter') {
                  handleSearch();
                }
              }}
              InputProps={{
                startAdornment: (
                  <InputAdornment position="start">
                    <SearchIcon />
                  </InputAdornment>
                ),
              }}
            />
            <Button
              variant="contained"
              startIcon={<SearchIcon />}
              onClick={handleSearch}
              sx={{ minWidth: 120 }}
            >
              Найти
            </Button>
          </Box>

          {/* Filters Toggle */}
          <Button
            startIcon={<FilterListIcon />}
            endIcon={filtersExpanded ? <ExpandLessIcon /> : <ExpandMoreIcon />}
            onClick={() => setFiltersExpanded(!filtersExpanded)}
            size="small"
          >
            Фильтры {hasActiveFilters && `(${1 + (dateFrom ? 1 : 0) + (dateTo ? 1 : 0)})`}
          </Button>

          {/* Filters Section */}
          <Collapse in={filtersExpanded}>
            <Box sx={{ mt: 2, pt: 2, borderTop: 1, borderColor: 'divider' }}>
              <Grid container spacing={2}>
                <Grid size={{ xs: 12, sm: 6, md: 4 }}>
                  <FormControl fullWidth size="small">
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
                </Grid>

                <Grid size={{ xs: 12, sm: 6, md: 4 }}>
                  <TextField
                    fullWidth
                    size="small"
                    type="date"
                    label="Дата от"
                    value={dateFrom}
                    onChange={(e) => setDateFrom(e.target.value)}
                    InputLabelProps={{ shrink: true }}
                  />
                </Grid>

                <Grid size={{ xs: 12, sm: 6, md: 4 }}>
                  <TextField
                    fullWidth
                    size="small"
                    type="date"
                    label="Дата до"
                    value={dateTo}
                    onChange={(e) => setDateTo(e.target.value)}
                    InputLabelProps={{ shrink: true }}
                  />
                </Grid>
              </Grid>

              <Box sx={{ mt: 2, display: 'flex', gap: 2 }}>
                <Button size="small" onClick={handleSearch} variant="outlined">
                  Применить фильтры
                </Button>
                {hasActiveFilters && (
                  <Button size="small" onClick={handleClearFilters}>
                    Сбросить фильтры
                  </Button>
                )}
              </Box>
            </Box>
          </Collapse>
        </Paper>

        {/* Results Counter */}
        {debouncedSearch && (
          <Box sx={{ mb: 2 }}>
            <Typography variant="body2" color="text.secondary">
              Найдено результатов: {searchResults.length}
            </Typography>
          </Box>
        )}

        {/* Results */}
        {!debouncedSearch && searchResults.length === 0 ? (
          <Box sx={{ textAlign: 'center', py: 8 }}>
            <SearchIcon sx={{ fontSize: 64, color: 'text.disabled', mb: 2 }} />
            <Typography variant="h6" color="text.secondary" gutterBottom>
              Введите поисковый запрос
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Начните вводить текст для поиска обращений
            </Typography>
          </Box>
        ) : searchResults.length === 0 ? (
          <Box sx={{ textAlign: 'center', py: 8 }}>
            <SearchIcon sx={{ fontSize: 64, color: 'text.disabled', mb: 2 }} />
            <Typography variant="h6" color="text.secondary" gutterBottom>
              Ничего не найдено
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Попробуйте изменить поисковый запрос или фильтры
            </Typography>
          </Box>
        ) : (
          <Box>
            {searchResults.map((issue) => (
              <Card
                key={issue.id}
                sx={{
                  mb: 2,
                  '&:hover': {
                    boxShadow: 3,
                  },
                }}
              >
                <CardActionArea onClick={() => handleIssueClick(issue)}>
                  <CardContent>
                    <Box
                      sx={{
                        display: 'flex',
                        justifyContent: 'space-between',
                        alignItems: 'flex-start',
                        mb: 1,
                      }}
                    >
                      <Box sx={{ flexGrow: 1 }}>
                        <Typography variant="h6" gutterBottom>
                          {issue.title || 'Без названия'}
                        </Typography>
                        <Typography
                          variant="body2"
                          color="text.secondary"
                          sx={{ mb: 1 }}
                        >
                          {truncateText(issue.description, 200)}
                        </Typography>
                      </Box>
                      {issue.status && (
                        <Chip
                          label={getStatusLabel(issue.status)}
                          size="small"
                          sx={{
                            backgroundColor: STATUS_COLORS[issue.status],
                            color: 'white',
                            ml: 2,
                          }}
                        />
                      )}
                    </Box>

                    <Divider sx={{ my: 1 }} />

                    <Box
                      sx={{
                        display: 'flex',
                        gap: 2,
                        alignItems: 'center',
                      }}
                    >
                      <Typography variant="caption" color="text.secondary">
                        ID: {issue.external_id}
                      </Typography>
                      <Typography variant="caption" color="text.secondary">
                        •
                      </Typography>
                      <Typography variant="caption" color="text.secondary">
                        Создано: {formatDate(issue.created_at, 'dd.MM.yyyy')}
                      </Typography>
                      {issue.priority && (
                        <>
                          <Typography variant="caption" color="text.secondary">
                            •
                          </Typography>
                          <Typography variant="caption" color="text.secondary">
                            Приоритет: {issue.priority}
                          </Typography>
                        </>
                      )}
                    </Box>
                  </CardContent>
                </CardActionArea>
              </Card>
            ))}
          </Box>
        )}
      </Box>
    </Container>
  );
};

export default Search;
