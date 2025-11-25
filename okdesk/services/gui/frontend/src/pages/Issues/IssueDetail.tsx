import React, { useMemo } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import {
  Container,
  Box,
  Typography,
  Paper,
  Card,
  CardContent,
  Chip,
  Button,
  Divider,
  Avatar,
  Grid,
} from '@mui/material';
import Timeline from '@mui/lab/Timeline';
import TimelineItem from '@mui/lab/TimelineItem';
import TimelineSeparator from '@mui/lab/TimelineSeparator';
import TimelineConnector from '@mui/lab/TimelineConnector';
import TimelineContent from '@mui/lab/TimelineContent';
import TimelineDot from '@mui/lab/TimelineDot';
import TimelineOppositeContent from '@mui/lab/TimelineOppositeContent';
import ArrowBackIcon from '@mui/icons-material/ArrowBack';
import PersonIcon from '@mui/icons-material/Person';
import SupportAgentIcon from '@mui/icons-material/SupportAgent';
import { mockIssues, mockMessages } from '@/api/mocks';
import type { AuthorType } from '@/api/types';
import { formatDate, formatRelativeTime } from '@/utils/formatters';
import { STATUS_COLORS, PRIORITY_COLORS } from '@/utils/constants';
import ConfidenceBar from '@/components/common/ConfidenceBar';
import TagBadge from '@/components/common/TagBadge';
import ErrorMessage from '@/components/common/ErrorMessage';

const IssueDetail: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();

  // Find issue and its messages
  const issue = useMemo(() => mockIssues.find((i) => i.id === id), [id]);
  const messages = useMemo(
    () => mockMessages.filter((m) => m.issue_id === id),
    [id]
  );

  if (!issue) {
    return (
      <Container maxWidth="xl">
        <Box sx={{ py: 4 }}>
          <ErrorMessage
            title="Обращение не найдено"
            message="Обращение с указанным ID не существует"
          />
          <Button
            startIcon={<ArrowBackIcon />}
            onClick={() => navigate('/issues')}
            sx={{ mt: 2 }}
          >
            Вернуться к списку
          </Button>
        </Box>
      </Container>
    );
  }

  const getStatusLabel = (status: string | undefined): string => {
    if (!status) return '-';
    const labels: Record<string, string> = {
      opened: 'Открыто',
      wait: 'Ожидание',
      completed: 'Завершено',
      closed: 'Закрыто',
    };
    return labels[status] || status;
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

  const getAuthorIcon = (authorType: AuthorType | undefined) => {
    if (authorType === 'employee') {
      return <SupportAgentIcon />;
    }
    return <PersonIcon />;
  };

  const getAuthorColor = (authorType: AuthorType | undefined) => {
    if (authorType === 'employee') {
      return 'primary.main';
    }
    return 'secondary.main';
  };

  return (
    <Container maxWidth="xl">
      <Box sx={{ py: 4 }}>
        {/* Back Button */}
        <Button
          startIcon={<ArrowBackIcon />}
          onClick={() => navigate('/issues')}
          sx={{ mb: 3 }}
        >
          Назад к списку
        </Button>

        {/* Issue Card */}
        <Card sx={{ mb: 3 }}>
          <CardContent>
            <Box sx={{ display: 'flex', alignItems: 'flex-start', gap: 2, mb: 2 }}>
              <Typography variant="h4" sx={{ flexGrow: 1 }}>
                {issue.title || 'Без названия'}
              </Typography>
              {issue.status && (
                <Chip
                  label={getStatusLabel(issue.status)}
                  sx={{
                    backgroundColor: STATUS_COLORS[issue.status],
                    color: 'white',
                  }}
                />
              )}
            </Box>

            <Grid container spacing={2} sx={{ mb: 2 }}>
              <Grid size={{ xs: 12, sm: 6, md: 3 }}>
                <Typography variant="caption" color="text.secondary">
                  Внешний ID
                </Typography>
                <Typography variant="body1" fontFamily="monospace">
                  {issue.external_id}
                </Typography>
              </Grid>

              {issue.priority && (
                <Grid size={{ xs: 12, sm: 6, md: 3 }}>
                  <Typography variant="caption" color="text.secondary">
                    Приоритет
                  </Typography>
                  <Box>
                    <Chip
                      label={getPriorityLabel(issue.priority)}
                      size="small"
                      sx={{
                        backgroundColor: PRIORITY_COLORS[issue.priority as keyof typeof PRIORITY_COLORS],
                        color: 'white',
                        mt: 0.5,
                      }}
                    />
                  </Box>
                </Grid>
              )}

              <Grid size={{ xs: 12, sm: 6, md: 3 }}>
                <Typography variant="caption" color="text.secondary">
                  Создано
                </Typography>
                <Typography variant="body2">
                  {formatDate(issue.created_at)}
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  {formatRelativeTime(issue.created_at)}
                </Typography>
              </Grid>

              <Grid size={{ xs: 12, sm: 6, md: 3 }}>
                <Typography variant="caption" color="text.secondary">
                  Обновлено
                </Typography>
                <Typography variant="body2">
                  {formatDate(issue.updated_at)}
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  {formatRelativeTime(issue.updated_at)}
                </Typography>
              </Grid>
            </Grid>

            {issue.description && (
              <>
                <Divider sx={{ my: 2 }} />
                <Typography variant="subtitle2" gutterBottom>
                  Описание
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  {issue.description}
                </Typography>
              </>
            )}
          </CardContent>
        </Card>

        {/* Messages Timeline */}
        <Paper sx={{ p: 3 }}>
          <Typography variant="h5" gutterBottom>
            История сообщений
          </Typography>

          {messages.length === 0 ? (
            <Box sx={{ textAlign: 'center', py: 4 }}>
              <Typography color="text.secondary">
                Нет сообщений в этом обращении
              </Typography>
            </Box>
          ) : (
            <Timeline position="right">
              {messages.map((message, index) => (
                <TimelineItem key={message.id}>
                  <TimelineOppositeContent color="text.secondary" sx={{ flex: 0.3 }}>
                    <Typography variant="body2">
                      {formatDate(message.published_at, 'dd.MM.yyyy')}
                    </Typography>
                    <Typography variant="caption">
                      {formatDate(message.published_at, 'HH:mm')}
                    </Typography>
                  </TimelineOppositeContent>

                  <TimelineSeparator>
                    <TimelineDot sx={{ bgcolor: getAuthorColor(message.author_type) }}>
                      <Avatar
                        sx={{
                          width: 32,
                          height: 32,
                          bgcolor: getAuthorColor(message.author_type),
                        }}
                      >
                        {getAuthorIcon(message.author_type)}
                      </Avatar>
                    </TimelineDot>
                    {index < messages.length - 1 && <TimelineConnector />}
                  </TimelineSeparator>

                  <TimelineContent>
                    <Card variant="outlined" sx={{ mb: 2 }}>
                      <CardContent>
                        {/* Author */}
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
                          <Typography variant="subtitle2" fontWeight="bold">
                            {message.author_name || 'Неизвестный'}
                          </Typography>
                          <Chip
                            label={
                              message.author_type === 'employee'
                                ? 'Сотрудник'
                                : message.author_type === 'contact'
                                ? 'Контакт'
                                : 'Пользователь'
                            }
                            size="small"
                            color={message.author_type === 'employee' ? 'primary' : 'default'}
                          />
                        </Box>

                        {/* Content */}
                        <Typography variant="body1" sx={{ mb: 2 }}>
                          {message.content}
                        </Typography>

                        {/* Analysis */}
                        {message.analysis && (
                          <Box sx={{ mt: 2, pt: 2, borderTop: 1, borderColor: 'divider' }}>
                            {/* Intents */}
                            {message.analysis.intents.length > 0 && (
                              <Box sx={{ mb: 2 }}>
                                <Typography variant="subtitle2" gutterBottom>
                                  Намерения:
                                </Typography>
                                {message.analysis.intents.map((intentItem) => (
                                  <Box key={intentItem.id} sx={{ mb: 1 }}>
                                    <Box
                                      sx={{
                                        display: 'flex',
                                        alignItems: 'center',
                                        gap: 1,
                                        mb: 0.5,
                                      }}
                                    >
                                      <Typography variant="body2" fontWeight="medium">
                                        {intentItem.intent?.name}
                                      </Typography>
                                      {intentItem.intent?.description && (
                                        <Typography
                                          variant="caption"
                                          color="text.secondary"
                                        >
                                          - {intentItem.intent.description}
                                        </Typography>
                                      )}
                                    </Box>
                                    <ConfidenceBar
                                      confidence={intentItem.confidence}
                                      tooltip={`Уверенность: ${Math.round(intentItem.confidence * 100)}%`}
                                    />
                                  </Box>
                                ))}
                              </Box>
                            )}

                            {/* Tags */}
                            {message.analysis.tags.length > 0 && (
                              <Box sx={{ mb: 2 }}>
                                <Typography variant="subtitle2" gutterBottom>
                                  Теги:
                                </Typography>
                                <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                                  {message.analysis.tags.map((tagItem) => (
                                    <TagBadge
                                      key={tagItem.id}
                                      name={tagItem.tag?.name || ''}
                                      type={tagItem.tag?.type}
                                      tooltip={`Уверенность: ${Math.round(tagItem.confidence * 100)}%`}
                                    />
                                  ))}
                                </Box>
                              </Box>
                            )}

                            {/* Reasoning */}
                            {message.analysis.reasoning && (
                              <Box>
                                <Typography variant="subtitle2" gutterBottom>
                                  Обоснование:
                                </Typography>
                                <Typography
                                  variant="body2"
                                  color="text.secondary"
                                  sx={{ fontStyle: 'italic' }}
                                >
                                  {message.analysis.reasoning}
                                </Typography>
                              </Box>
                            )}
                          </Box>
                        )}
                      </CardContent>
                    </Card>
                  </TimelineContent>
                </TimelineItem>
              ))}
            </Timeline>
          )}
        </Paper>
      </Box>
    </Container>
  );
};

export default IssueDetail;
