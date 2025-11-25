import React, { useState } from 'react';
import {
  Container,
  Box,
  Typography,
  Card,
  CardContent,
  CardActionArea,
  Chip,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  Divider,
  Grid,
} from '@mui/material';
import CategoryIcon from '@mui/icons-material/Category';
import { mockClusters, mockIssues } from '@/api/mocks';
import type { Cluster, Issue } from '@/api/types';
import { formatNumber, formatDate } from '@/utils/formatters';
import { CHART_COLORS } from '@/utils/constants';

const Clusters: React.FC = () => {
  const [selectedCluster, setSelectedCluster] = useState<Cluster | null>(null);
  const [detailModalOpen, setDetailModalOpen] = useState(false);

  const handleClusterClick = (cluster: Cluster) => {
    setSelectedCluster(cluster);
    setDetailModalOpen(true);
  };

  const handleCloseModal = () => {
    setDetailModalOpen(false);
    setSelectedCluster(null);
  };

  // Get cluster color by index
  const getClusterColor = (index: number): string => {
    return CHART_COLORS[index % CHART_COLORS.length];
  };

  // Mock function to get issues in a cluster (in real app, would fetch from API)
  const getClusterIssues = (_clusterId: string): Issue[] => {
    // For demo, return first 5 issues
    return mockIssues.slice(0, Math.min(5, mockIssues.length));
  };

  return (
    <Container maxWidth="xl">
      <Box sx={{ py: 4 }}>
        <Typography variant="h4" gutterBottom>
          Кластеры обращений
        </Typography>
        <Typography variant="body1" color="text.secondary" paragraph>
          Автоматически сгруппированные обращения по схожести тематики
        </Typography>

        {/* Summary */}
        <Card sx={{ mb: 4, bgcolor: 'primary.main', color: 'white' }}>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              Всего кластеров: {mockClusters.length}
            </Typography>
            <Typography variant="body2">
              Суммарно обращений в кластерах:{' '}
              {formatNumber(mockClusters.reduce((sum, c) => sum + c.size, 0))}
            </Typography>
          </CardContent>
        </Card>

        {/* Cluster Grid */}
        <Grid container spacing={3}>
          {mockClusters.map((cluster, index) => (
            <Grid size={{ xs: 12, sm: 6, md: 4 }} key={cluster.id}>
              <Card
                sx={{
                  height: '100%',
                  borderLeft: 4,
                  borderColor: getClusterColor(index),
                  transition: 'transform 0.2s, box-shadow 0.2s',
                  '&:hover': {
                    transform: 'translateY(-4px)',
                    boxShadow: 4,
                  },
                }}
              >
                <CardActionArea onClick={() => handleClusterClick(cluster)}>
                  <CardContent>
                    <Box
                      sx={{
                        display: 'flex',
                        alignItems: 'center',
                        gap: 1,
                        mb: 2,
                      }}
                    >
                      <CategoryIcon
                        sx={{ color: getClusterColor(index), fontSize: 32 }}
                      />
                      <Typography variant="h6" component="div">
                        {cluster.name || `Кластер ${cluster.cluster_label}`}
                      </Typography>
                    </Box>

                    {cluster.description && (
                      <Typography
                        variant="body2"
                        color="text.secondary"
                        sx={{ mb: 2, minHeight: 60 }}
                      >
                        {cluster.description}
                      </Typography>
                    )}

                    <Box
                      sx={{
                        display: 'flex',
                        justifyContent: 'space-between',
                        alignItems: 'center',
                      }}
                    >
                      <Chip
                        label={`${formatNumber(cluster.size)} обращений`}
                        size="small"
                        sx={{
                          bgcolor: getClusterColor(index),
                          color: 'white',
                        }}
                      />
                      <Typography variant="caption" color="text.secondary">
                        Кластер #{cluster.cluster_label}
                      </Typography>
                    </Box>
                  </CardContent>
                </CardActionArea>
              </Card>
            </Grid>
          ))}
        </Grid>

        {/* Empty State */}
        {mockClusters.length === 0 && (
          <Box sx={{ textAlign: 'center', py: 8 }}>
            <CategoryIcon sx={{ fontSize: 64, color: 'text.disabled', mb: 2 }} />
            <Typography variant="h6" color="text.secondary" gutterBottom>
              Кластеры не найдены
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Выполните кластеризацию обращений для создания групп
            </Typography>
          </Box>
        )}
      </Box>

      {/* Cluster Detail Modal */}
      <Dialog
        open={detailModalOpen}
        onClose={handleCloseModal}
        maxWidth="md"
        fullWidth
      >
        {selectedCluster && (
          <>
            <DialogTitle>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                <CategoryIcon
                  sx={{
                    color: getClusterColor(
                      mockClusters.findIndex((c) => c.id === selectedCluster.id)
                    ),
                    fontSize: 32,
                  }}
                />
                <Box>
                  <Typography variant="h6">
                    {selectedCluster.name || `Кластер ${selectedCluster.cluster_label}`}
                  </Typography>
                  <Typography variant="caption" color="text.secondary">
                    Кластер #{selectedCluster.cluster_label}
                  </Typography>
                </Box>
              </Box>
            </DialogTitle>
            <DialogContent dividers>
              {selectedCluster.description && (
                <Box sx={{ mb: 3 }}>
                  <Typography variant="subtitle2" gutterBottom>
                    Описание
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    {selectedCluster.description}
                  </Typography>
                </Box>
              )}

              <Box sx={{ mb: 3 }}>
                <Typography variant="subtitle2" gutterBottom>
                  Статистика
                </Typography>
                <Grid container spacing={2}>
                  <Grid size={{ xs: 6 }}>
                    <Typography variant="caption" color="text.secondary">
                      Количество обращений
                    </Typography>
                    <Typography variant="h6">
                      {formatNumber(selectedCluster.size)}
                    </Typography>
                  </Grid>
                  <Grid size={{ xs: 6 }}>
                    <Typography variant="caption" color="text.secondary">
                      Последнее обновление
                    </Typography>
                    <Typography variant="body2">
                      {formatDate(selectedCluster.updated_at)}
                    </Typography>
                  </Grid>
                </Grid>
              </Box>

              <Divider sx={{ my: 2 }} />

              <Typography variant="subtitle2" gutterBottom>
                Примеры обращений в кластере
              </Typography>

              {getClusterIssues(selectedCluster.id).map((issue) => (
                <Card key={issue.id} variant="outlined" sx={{ mb: 1 }}>
                  <CardContent sx={{ py: 1.5 }}>
                    <Typography variant="body2" fontWeight="medium" gutterBottom>
                      {issue.title || 'Без названия'}
                    </Typography>
                    {issue.description && (
                      <Typography
                        variant="caption"
                        color="text.secondary"
                        sx={{
                          display: '-webkit-box',
                          WebkitLineClamp: 2,
                          WebkitBoxOrient: 'vertical',
                          overflow: 'hidden',
                        }}
                      >
                        {issue.description}
                      </Typography>
                    )}
                    <Box
                      sx={{
                        display: 'flex',
                        gap: 1,
                        mt: 1,
                        alignItems: 'center',
                      }}
                    >
                      <Typography variant="caption" color="text.secondary">
                        {issue.external_id}
                      </Typography>
                      <Typography variant="caption" color="text.secondary">
                        •
                      </Typography>
                      <Typography variant="caption" color="text.secondary">
                        {formatDate(issue.created_at, 'dd.MM.yyyy')}
                      </Typography>
                    </Box>
                  </CardContent>
                </Card>
              ))}

              {selectedCluster.size > 5 && (
                <Typography variant="caption" color="text.secondary" sx={{ mt: 1 }}>
                  И еще {selectedCluster.size - 5} обращений...
                </Typography>
              )}
            </DialogContent>
            <DialogActions>
              <Button onClick={handleCloseModal}>Закрыть</Button>
            </DialogActions>
          </>
        )}
      </Dialog>
    </Container>
  );
};

export default Clusters;
