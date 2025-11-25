import React from 'react';
import { Box, Container, Typography, Paper } from '@mui/material';
import ConstructionIcon from '@mui/icons-material/Construction';

const ServiceUnavailable: React.FC = () => {
  return (
    <Container maxWidth="md" sx={{ py: 8 }}>
      <Paper
        elevation={3}
        sx={{
          p: 6,
          textAlign: 'center',
          borderRadius: 2,
        }}
      >
        <Box
          sx={{
            display: 'flex',
            justifyContent: 'center',
            mb: 3,
          }}
        >
          <ConstructionIcon
            sx={{
              fontSize: 80,
              color: 'warning.main',
            }}
          />
        </Box>

        <Typography variant="h4" gutterBottom fontWeight="bold">
          Сервис недоступен
        </Typography>

        <Typography variant="body1" color="text.secondary" paragraph>
          GUI-сервис находится в разработке и пока недоступен.
        </Typography>

        <Typography variant="body2" color="text.secondary">
          API Gateway и backend-сервисы ещё не реализованы.
          Следите за обновлениями в документации проекта.
        </Typography>
      </Paper>
    </Container>
  );
};

export default ServiceUnavailable;
