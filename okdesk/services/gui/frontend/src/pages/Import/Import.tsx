import React, { useState } from 'react';
import {
  Container,
  Box,
  Typography,
  Tabs,
  Tab,
  Paper,
  RadioGroup,
  FormControlLabel,
  Radio,
  Button,
  Chip,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Divider,
  Grid,
} from '@mui/material';
import FileUpload from '@/components/upload/FileUpload';
import DataTable, { type Column } from '@/components/tables/DataTable';
import ErrorMessage from '@/components/common/ErrorMessage';
import { mockImports } from '@/api/mocks';
import type { Import, SourceType, ImportStatus } from '@/api/types';
import { formatDate, formatDuration, formatNumber } from '@/utils/formatters';
import { IMPORT_STATUS_COLORS, ACCEPTED_FILE_TYPES } from '@/utils/constants';

const ImportPage: React.FC = () => {
  const [activeTab, setActiveTab] = useState(0);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [sourceType, setSourceType] = useState<SourceType>('okdesk');
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedImport, setSelectedImport] = useState<Import | null>(null);
  const [detailModalOpen, setDetailModalOpen] = useState(false);

  const handleFileSelect = (file: File) => {
    setSelectedFile(file);
    setError(null);
  };

  const handleSourceTypeChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setSourceType(event.target.value as SourceType);
    setSelectedFile(null);
  };

  const handleUpload = async () => {
    if (!selectedFile) {
      setError('Выберите файл для загрузки');
      return;
    }

    setUploading(true);
    setError(null);

    // Mock upload - in real app, call API here
    setTimeout(() => {
      setUploading(false);
      setSelectedFile(null);
      alert('Файл успешно загружен! Импорт начат.');
      setActiveTab(1); // Switch to history tab
    }, 2000);
  };

  const getStatusColor = (status: ImportStatus): string => {
    return IMPORT_STATUS_COLORS[status];
  };

  const getStatusLabel = (status: ImportStatus): string => {
    const labels: Record<ImportStatus, string> = {
      in_progress: 'В процессе',
      completed: 'Завершен',
      failed: 'Ошибка',
    };
    return labels[status];
  };

  const columns: Column<Import>[] = [
    {
      id: 'filename',
      label: 'Имя файла',
      sortable: true,
      render: (row) => row.filename || '-',
    },
    {
      id: 'status',
      label: 'Статус',
      sortable: true,
      render: (row) => (
        <Chip
          label={getStatusLabel(row.status)}
          size="small"
          sx={{
            backgroundColor: getStatusColor(row.status),
            color: 'white',
          }}
        />
      ),
    },
    {
      id: 'started_at',
      label: 'Начало',
      sortable: true,
      render: (row) => formatDate(row.started_at),
    },
    {
      id: 'completed_at',
      label: 'Окончание',
      sortable: true,
      render: (row) => formatDate(row.completed_at),
    },
    {
      id: 'stats',
      label: 'Обращений',
      align: 'right',
      render: (row) => formatNumber(row.stats?.total_issues),
    },
    {
      id: 'messages',
      label: 'Сообщений',
      align: 'right',
      render: (row) => formatNumber(row.stats?.total_messages),
    },
  ];

  return (
    <Container maxWidth="xl">
      <Box sx={{ py: 4 }}>
        <Typography variant="h4" gutterBottom>
          Импорт данных
        </Typography>
        <Typography variant="body1" color="text.secondary" paragraph>
          Загрузите файлы с данными из OKDesk или Telegram для анализа
        </Typography>

        <Paper sx={{ mt: 3 }}>
          <Tabs
            value={activeTab}
            onChange={(_, newValue) => setActiveTab(newValue)}
            sx={{ borderBottom: 1, borderColor: 'divider' }}
          >
            <Tab label="Загрузка файла" />
            <Tab label="История импортов" />
          </Tabs>

          {/* Tab 1: File Upload */}
          {activeTab === 0 && (
            <Box sx={{ p: 3 }}>
              <Typography variant="h6" gutterBottom>
                Тип источника
              </Typography>
              <RadioGroup
                row
                value={sourceType}
                onChange={handleSourceTypeChange}
                sx={{ mb: 3 }}
              >
                <FormControlLabel
                  value="okdesk"
                  control={<Radio />}
                  label="OKDesk (JSONL)"
                />
                <FormControlLabel
                  value="telegram"
                  control={<Radio />}
                  label="Telegram (JSON)"
                />
              </RadioGroup>

              <FileUpload
                onFileSelect={handleFileSelect}
                acceptedTypes={ACCEPTED_FILE_TYPES[sourceType]}
                disabled={uploading}
                uploading={uploading}
                error={error || undefined}
              />

              <Box sx={{ mt: 3, display: 'flex', gap: 2 }}>
                <Button
                  variant="contained"
                  size="large"
                  onClick={handleUpload}
                  disabled={!selectedFile || uploading}
                >
                  Начать импорт
                </Button>
                {selectedFile && !uploading && (
                  <Button
                    variant="outlined"
                    size="large"
                    onClick={() => {
                      setSelectedFile(null);
                      setError(null);
                    }}
                  >
                    Отменить
                  </Button>
                )}
              </Box>

              {error && <ErrorMessage message={error} />}
            </Box>
          )}

          {/* Tab 2: Import History */}
          {activeTab === 1 && (
            <Box sx={{ p: 3 }}>
              <Box
                sx={{
                  '& table tbody tr': {
                    cursor: 'pointer',
                    '&:hover': {
                      backgroundColor: 'action.hover',
                    },
                  },
                }}
                onClick={(e) => {
                  // Find the clicked row and get import ID
                  const target = e.target as HTMLElement;
                  const row = target.closest('tr');
                  if (row && row.dataset.id) {
                    const imp = mockImports.find((i) => i.id === row.dataset.id);
                    if (imp) {
                      setSelectedImport(imp);
                      setDetailModalOpen(true);
                    }
                  }
                }}
              >
                <DataTable
                  columns={columns}
                  rows={mockImports}
                  getRowId={(row) => row.id}
                  emptyMessage="История импортов пуста"
                />
              </Box>

              <Box sx={{ mt: 2 }}>
                <Typography variant="caption" color="text.secondary">
                  Нажмите на строку для просмотра деталей импорта
                </Typography>
              </Box>
            </Box>
          )}
        </Paper>
      </Box>

      {/* Import Details Modal */}
      <Dialog
        open={detailModalOpen}
        onClose={() => setDetailModalOpen(false)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>
          Детали импорта
          {selectedImport && (
            <Chip
              label={getStatusLabel(selectedImport.status)}
              size="small"
              sx={{
                ml: 2,
                backgroundColor: getStatusColor(selectedImport.status),
                color: 'white',
              }}
            />
          )}
        </DialogTitle>
        <DialogContent dividers>
          {selectedImport && (
            <Grid container spacing={2}>
              <Grid size={{ xs: 12 }}>
                <Typography variant="subtitle2" color="text.secondary">
                  Имя файла
                </Typography>
                <Typography variant="body1">
                  {selectedImport.filename || '-'}
                </Typography>
              </Grid>

              <Grid size={{ xs: 6 }}>
                <Typography variant="subtitle2" color="text.secondary">
                  Начало
                </Typography>
                <Typography variant="body1">
                  {formatDate(selectedImport.started_at)}
                </Typography>
              </Grid>

              <Grid size={{ xs: 6 }}>
                <Typography variant="subtitle2" color="text.secondary">
                  Окончание
                </Typography>
                <Typography variant="body1">
                  {formatDate(selectedImport.completed_at)}
                </Typography>
              </Grid>

              {selectedImport.stats && (
                <>
                  <Grid size={{ xs: 12 }}>
                    <Divider sx={{ my: 1 }} />
                    <Typography variant="h6" gutterBottom>
                      Статистика
                    </Typography>
                  </Grid>

                  <Grid size={{ xs: 6 }}>
                    <Typography variant="subtitle2" color="text.secondary">
                      Всего обращений
                    </Typography>
                    <Typography variant="body1">
                      {formatNumber(selectedImport.stats.total_issues)}
                    </Typography>
                  </Grid>

                  <Grid size={{ xs: 6 }}>
                    <Typography variant="subtitle2" color="text.secondary">
                      Новых обращений
                    </Typography>
                    <Typography variant="body1">
                      {formatNumber(selectedImport.stats.new_issues)}
                    </Typography>
                  </Grid>

                  {selectedImport.stats.updated_issues !== undefined && (
                    <Grid size={{ xs: 6 }}>
                      <Typography variant="subtitle2" color="text.secondary">
                        Обновлено обращений
                      </Typography>
                      <Typography variant="body1">
                        {formatNumber(selectedImport.stats.updated_issues)}
                      </Typography>
                    </Grid>
                  )}

                  <Grid size={{ xs: 6 }}>
                    <Typography variant="subtitle2" color="text.secondary">
                      Всего сообщений
                    </Typography>
                    <Typography variant="body1">
                      {formatNumber(selectedImport.stats.total_messages)}
                    </Typography>
                  </Grid>

                  <Grid size={{ xs: 6 }}>
                    <Typography variant="subtitle2" color="text.secondary">
                      Новых сообщений
                    </Typography>
                    <Typography variant="body1">
                      {formatNumber(selectedImport.stats.new_messages)}
                    </Typography>
                  </Grid>

                  <Grid size={{ xs: 6 }}>
                    <Typography variant="subtitle2" color="text.secondary">
                      Длительность
                    </Typography>
                    <Typography variant="body1">
                      {formatDuration(selectedImport.stats.duration_seconds)}
                    </Typography>
                  </Grid>
                </>
              )}

              {selectedImport.error_message && (
                <Grid size={{ xs: 12 }}>
                  <Divider sx={{ my: 1 }} />
                  <Typography variant="subtitle2" color="error">
                    Ошибка
                  </Typography>
                  <Typography variant="body2" color="error">
                    {selectedImport.error_message}
                  </Typography>
                </Grid>
              )}
            </Grid>
          )}
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDetailModalOpen(false)}>Закрыть</Button>
        </DialogActions>
      </Dialog>
    </Container>
  );
};

export default ImportPage;
