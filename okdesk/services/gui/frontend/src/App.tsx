import React from 'react';
import { Box, Toolbar } from '@mui/material';
import { ThemeProvider } from './contexts/ThemeContext';
import Header from './components/common/Header';
import ErrorBoundary from './components/common/ErrorBoundary';
import ServiceUnavailable from './pages/ServiceUnavailable';

const App: React.FC = () => {
  return (
    <ThemeProvider>
      <ErrorBoundary>
        <Box sx={{ display: 'flex' }}>
          <Header onMenuClick={() => {}} />

          <Box
            component="main"
            sx={{
              flexGrow: 1,
              p: 3,
              width: '100%',
            }}
          >
            <Toolbar />
            <ServiceUnavailable />
          </Box>
        </Box>
      </ErrorBoundary>
    </ThemeProvider>
  );
};

export default App;
