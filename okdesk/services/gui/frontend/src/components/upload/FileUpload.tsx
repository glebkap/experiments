import React, { useCallback, useState } from 'react';
import {
  Box,
  Button,
  Typography,
  LinearProgress,
  Alert,
  Paper,
} from '@mui/material';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import { formatFileSize } from '@/utils/formatters';

interface FileUploadProps {
  onFileSelect: (file: File) => void;
  acceptedTypes?: string[];
  maxSize?: number;
  disabled?: boolean;
  uploading?: boolean;
  uploadProgress?: number;
  error?: string;
}

const FileUpload: React.FC<FileUploadProps> = ({
  onFileSelect,
  acceptedTypes = ['.json', '.jsonl'],
  maxSize = 100 * 1024 * 1024, // 100 MB
  disabled = false,
  uploading = false,
  uploadProgress,
  error,
}) => {
  const [dragActive, setDragActive] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [validationError, setValidationError] = useState<string | null>(null);

  const validateFile = (file: File): string | null => {
    // Check file size
    if (file.size > maxSize) {
      return `Файл слишком большой. Максимальный размер: ${formatFileSize(maxSize)}`;
    }

    // Check file type
    if (acceptedTypes.length > 0) {
      const fileExtension = `.${file.name.split('.').pop()?.toLowerCase()}`;
      if (!acceptedTypes.includes(fileExtension)) {
        return `Недопустимый тип файла. Разрешены: ${acceptedTypes.join(', ')}`;
      }
    }

    return null;
  };

  const handleFile = useCallback(
    (file: File) => {
      const error = validateFile(file);
      if (error) {
        setValidationError(error);
        setSelectedFile(null);
        return;
      }

      setValidationError(null);
      setSelectedFile(file);
      onFileSelect(file);
    },
    [onFileSelect, maxSize, acceptedTypes],
  );

  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      e.stopPropagation();
      setDragActive(false);

      if (disabled || uploading) return;

      if (e.dataTransfer.files && e.dataTransfer.files[0]) {
        handleFile(e.dataTransfer.files[0]);
      }
    },
    [disabled, uploading, handleFile],
  );

  const handleFileInputChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      if (disabled || uploading) return;

      if (e.target.files && e.target.files[0]) {
        handleFile(e.target.files[0]);
      }
    },
    [disabled, uploading, handleFile],
  );

  const displayError = validationError || error;

  return (
    <Box>
      <Paper
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
        elevation={dragActive ? 8 : 1}
        sx={{
          border: 2,
          borderStyle: 'dashed',
          borderColor: dragActive
            ? 'primary.main'
            : displayError
            ? 'error.main'
            : 'grey.300',
          backgroundColor: dragActive
            ? 'action.hover'
            : disabled || uploading
            ? 'action.disabledBackground'
            : 'background.paper',
          padding: 4,
          textAlign: 'center',
          cursor: disabled || uploading ? 'not-allowed' : 'pointer',
          transition: 'all 0.2s ease',
        }}
      >
        <input
          type="file"
          id="file-upload-input"
          accept={acceptedTypes.join(',')}
          onChange={handleFileInputChange}
          disabled={disabled || uploading}
          style={{ display: 'none' }}
        />

        <label htmlFor="file-upload-input" style={{ cursor: 'inherit' }}>
          <Box
            display="flex"
            flexDirection="column"
            alignItems="center"
            gap={2}
          >
            <CloudUploadIcon
              sx={{
                fontSize: 64,
                color: dragActive
                  ? 'primary.main'
                  : disabled || uploading
                  ? 'action.disabled'
                  : 'text.secondary',
              }}
            />

            {uploading ? (
              <Box sx={{ width: '100%', maxWidth: 400 }}>
                <Typography variant="body1" gutterBottom>
                  Загрузка файла...
                </Typography>
                {uploadProgress !== undefined && (
                  <>
                    <LinearProgress
                      variant="determinate"
                      value={uploadProgress}
                      sx={{ mb: 1 }}
                    />
                    <Typography variant="body2" color="text.secondary">
                      {uploadProgress}%
                    </Typography>
                  </>
                )}
              </Box>
            ) : (
              <>
                <Typography variant="h6" color="text.primary">
                  {dragActive
                    ? 'Отпустите файл здесь'
                    : 'Перетащите файл сюда или нажмите для выбора'}
                </Typography>

                <Button
                  variant="contained"
                  component="span"
                  disabled={disabled}
                  sx={{ pointerEvents: 'none' }}
                >
                  Выбрать файл
                </Button>

                <Typography variant="body2" color="text.secondary">
                  Допустимые форматы: {acceptedTypes.join(', ')}
                  <br />
                  Максимальный размер: {formatFileSize(maxSize)}
                </Typography>
              </>
            )}

            {selectedFile && !uploading && (
              <Box sx={{ mt: 2 }}>
                <Typography variant="body2" color="text.secondary">
                  Выбран файл:
                </Typography>
                <Typography variant="body1" fontWeight="medium">
                  {selectedFile.name} ({formatFileSize(selectedFile.size)})
                </Typography>
              </Box>
            )}
          </Box>
        </label>
      </Paper>

      {displayError && (
        <Alert severity="error" sx={{ mt: 2 }}>
          {displayError}
        </Alert>
      )}
    </Box>
  );
};

export default FileUpload;
