import React, { useState, useMemo } from 'react';
import {
  Container,
  Box,
  Typography,
  Paper,
  RadioGroup,
  FormControlLabel,
  Radio,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  TextField,
  Button,
  Alert,
  Card,
  CardContent,
  Collapse,
  Grid,
} from '@mui/material';
import DownloadIcon from '@mui/icons-material/Download';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import ExpandLessIcon from '@mui/icons-material/ExpandLess';
import { mockIssues } from '@/api/mocks';
import type { IssueStatus } from '@/api/types';
import { formatNumber } from '@/utils/formatters';

type ExportFormat = 'csv' | 'json';

const Export: React.FC = () => {
  const [format, setFormat] = useState<ExportFormat>('csv');
  const [filtersExpanded, setFiltersExpanded] = useState(false);
  const [statusFilter, setStatusFilter] = useState<IssueStatus | 'all'>('all');
  const [dateFrom, setDateFrom] = useState('');
  const [dateTo, setDateTo] = useState('');
  const [includeMessages, setIncludeMessages] = useState(true);
  const [includeAnalysis, setIncludeAnalysis] = useState(true);
  const [exporting, setExporting] = useState(false);

  // Calculate preview count based on filters
  const previewCount = useMemo(() => {
    let filtered = mockIssues;

    if (statusFilter !== 'all') {
      filtered = filtered.filter((issue) => issue.status === statusFilter);
    }

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

    return filtered.length;
  }, [statusFilter, dateFrom, dateTo]);

  const handleExport = async () => {
    setExporting(true);

    // Mock export - in real app, would call API here
    setTimeout(() => {
      setExporting(false);
      alert(
        `Экспорт выполнен!\nФормат: ${format.toUpperCase()}\nОбращений: ${previewCount}\nСообщения: ${includeMessages ? 'Да' : 'Нет'}\nАнализ: ${includeAnalysis ? 'Да' : 'Нет'}`
      );
    }, 2000);
  };

  const handleClearFilters = () => {
    setStatusFilter('all');
    setDateFrom('');
    setDateTo('');
  };

  const hasActiveFilters = statusFilter !== 'all' || dateFrom || dateTo;

  const getFormatDescription = (fmt: ExportFormat): string => {
    switch (fmt) {
      case 'csv':
        return 'Табличный формат, совместимый с Excel, Google Sheets';
      case 'json':
        return 'Структурированный формат для программного использования';
      default:
        return '';
    }
  };

  return (
    <Container maxWidth="lg">
      <Box sx={{ py: 4 }}>
        <Typography variant="h4" gutterBottom>
          Экспорт данных
        </Typography>
        <Typography variant="body1" color="text.secondary" paragraph>
          Выгрузите обращения и результаты анализа в удобном формате
        </Typography>

        {/* Format Selection */}
        <Paper sx={{ p: 3, mb: 3 }}>
          <Typography variant="h6" gutterBottom>
            Формат экспорта
          </Typography>
          <RadioGroup
            value={format}
            onChange={(e) => setFormat(e.target.value as ExportFormat)}
          >
            <FormControlLabel
              value="csv"
              control={<Radio />}
              label={
                <Box>
                  <Typography variant="body1">CSV</Typography>
                  <Typography variant="caption" color="text.secondary">
                    {getFormatDescription('csv')}
                  </Typography>
                </Box>
              }
            />
            <FormControlLabel
              value="json"
              control={<Radio />}
              label={
                <Box>
                  <Typography variant="body1">JSON</Typography>
                  <Typography variant="caption" color="text.secondary">
                    {getFormatDescription('json')}
                  </Typography>
                </Box>
              }
            />
          </RadioGroup>
        </Paper>

        {/* Filters */}
        <Paper sx={{ p: 3, mb: 3 }}>
          <Box
            sx={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              mb: 2,
            }}
          >
            <Typography variant="h6">Фильтры</Typography>
            <Button
              size="small"
              endIcon={filtersExpanded ? <ExpandLessIcon /> : <ExpandMoreIcon />}
              onClick={() => setFiltersExpanded(!filtersExpanded)}
            >
              {filtersExpanded ? 'Скрыть' : 'Показать'}
            </Button>
          </Box>

          <Collapse in={filtersExpanded}>
            <Grid container spacing={2}>
              <Grid size={{ xs: 12, sm: 6, md: 4 }}>
                <FormControl fullWidth>
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
                  type="date"
                  label="Дата до"
                  value={dateTo}
                  onChange={(e) => setDateTo(e.target.value)}
                  InputLabelProps={{ shrink: true }}
                />
              </Grid>
            </Grid>

            {hasActiveFilters && (
              <Box sx={{ mt: 2 }}>
                <Button size="small" onClick={handleClearFilters}>
                  Сбросить фильтры
                </Button>
              </Box>
            )}
          </Collapse>
        </Paper>

        {/* Export Options */}
        <Paper sx={{ p: 3, mb: 3 }}>
          <Typography variant="h6" gutterBottom>
            Параметры экспорта
          </Typography>

          <FormControlLabel
            control={
              <Radio
                checked={includeMessages}
                onChange={(e) => setIncludeMessages(e.target.checked)}
              />
            }
            label={
              <Box>
                <Typography variant="body1">Включить сообщения</Typography>
                <Typography variant="caption" color="text.secondary">
                  Экспортировать все сообщения в обращениях
                </Typography>
              </Box>
            }
            sx={{ display: 'block', mb: 2 }}
          />

          <FormControlLabel
            control={
              <Radio
                checked={includeAnalysis}
                onChange={(e) => setIncludeAnalysis(e.target.checked)}
              />
            }
            label={
              <Box>
                <Typography variant="body1">Включить результаты анализа</Typography>
                <Typography variant="caption" color="text.secondary">
                  Экспортировать намерения, теги и обоснование
                </Typography>
              </Box>
            }
            sx={{ display: 'block' }}
          />
        </Paper>

        {/* Preview */}
        <Card sx={{ mb: 3, bgcolor: 'primary.light' }}>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              Предпросмотр экспорта
            </Typography>
            <Grid container spacing={2}>
              <Grid size={{ xs: 12, sm: 6 }}>
                <Typography variant="body2" color="text.secondary">
                  Формат
                </Typography>
                <Typography variant="h6">{format.toUpperCase()}</Typography>
              </Grid>
              <Grid size={{ xs: 12, sm: 6 }}>
                <Typography variant="body2" color="text.secondary">
                  Количество обращений
                </Typography>
                <Typography variant="h6">{formatNumber(previewCount)}</Typography>
              </Grid>
              <Grid size={{ xs: 12, sm: 6 }}>
                <Typography variant="body2" color="text.secondary">
                  Включить сообщения
                </Typography>
                <Typography variant="h6">{includeMessages ? 'Да' : 'Нет'}</Typography>
              </Grid>
              <Grid size={{ xs: 12, sm: 6 }}>
                <Typography variant="body2" color="text.secondary">
                  Включить анализ
                </Typography>
                <Typography variant="h6">{includeAnalysis ? 'Да' : 'Нет'}</Typography>
              </Grid>
            </Grid>
          </CardContent>
        </Card>

        {/* Export Button */}
        <Box sx={{ display: 'flex', gap: 2, justifyContent: 'center' }}>
          <Button
            variant="contained"
            size="large"
            startIcon={<DownloadIcon />}
            onClick={handleExport}
            disabled={exporting || previewCount === 0}
          >
            {exporting ? 'Экспорт...' : 'Скачать экспорт'}
          </Button>
        </Box>

        {previewCount === 0 && (
          <Alert severity="warning" sx={{ mt: 2 }}>
            Нет обращений, соответствующих выбранным фильтрам
          </Alert>
        )}

        {/* Info */}
        <Alert severity="info" sx={{ mt: 3 }}>
          <Typography variant="body2">
            <strong>Обратите внимание:</strong> Экспорт больших объемов данных может
            занять некоторое время. Файл будет автоматически загружен после завершения.
          </Typography>
        </Alert>
      </Box>
    </Container>
  );
};

export default Export;
