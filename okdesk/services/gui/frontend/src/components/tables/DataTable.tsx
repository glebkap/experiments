import React, { useState } from 'react';
import {
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  TablePagination,
  TableSortLabel,
  Checkbox,
  Paper,
} from '@mui/material';

export interface Column<T> {
  id: string;
  label: string;
  sortable?: boolean;
  render?: (row: T) => React.ReactNode;
  align?: 'left' | 'right' | 'center';
  width?: string | number;
}

interface DataTableProps<T> {
  columns: Column<T>[];
  rows: T[];
  getRowId: (row: T) => string;
  selectable?: boolean;
  selected?: string[];
  onSelectionChange?: (selected: string[]) => void;
  page?: number;
  rowsPerPage?: number;
  totalRows?: number;
  onPageChange?: (page: number) => void;
  onRowsPerPageChange?: (rowsPerPage: number) => void;
  rowsPerPageOptions?: number[];
  sortBy?: string;
  sortOrder?: 'asc' | 'desc';
  onSortChange?: (columnId: string) => void;
  emptyMessage?: string;
}

function DataTable<T>({
  columns,
  rows,
  getRowId,
  selectable = false,
  selected = [],
  onSelectionChange,
  page = 0,
  rowsPerPage = 20,
  totalRows,
  onPageChange,
  onRowsPerPageChange,
  rowsPerPageOptions = [10, 20, 50, 100],
  sortBy,
  sortOrder = 'asc',
  onSortChange,
  emptyMessage = 'Нет данных',
}: DataTableProps<T>) {
  const [localSortBy, setLocalSortBy] = useState<string | undefined>(sortBy);
  const [localSortOrder, setLocalSortOrder] = useState<'asc' | 'desc'>(sortOrder);

  const isSelected = (id: string) => selected.indexOf(id) !== -1;

  const handleSelectAll = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.checked) {
      const newSelected = rows.map(getRowId);
      onSelectionChange?.(newSelected);
    } else {
      onSelectionChange?.([]);
    }
  };

  const handleSelectRow = (id: string) => {
    const selectedIndex = selected.indexOf(id);
    let newSelected: string[] = [];

    if (selectedIndex === -1) {
      newSelected = newSelected.concat(selected, id);
    } else if (selectedIndex === 0) {
      newSelected = newSelected.concat(selected.slice(1));
    } else if (selectedIndex === selected.length - 1) {
      newSelected = newSelected.concat(selected.slice(0, -1));
    } else if (selectedIndex > 0) {
      newSelected = newSelected.concat(
        selected.slice(0, selectedIndex),
        selected.slice(selectedIndex + 1),
      );
    }

    onSelectionChange?.(newSelected);
  };

  const handleSort = (columnId: string) => {
    if (onSortChange) {
      onSortChange(columnId);
    } else {
      const isAsc = localSortBy === columnId && localSortOrder === 'asc';
      setLocalSortOrder(isAsc ? 'desc' : 'asc');
      setLocalSortBy(columnId);
    }
  };

  const handleChangePage = (_event: unknown, newPage: number) => {
    onPageChange?.(newPage);
  };

  const handleChangeRowsPerPage = (event: React.ChangeEvent<HTMLInputElement>) => {
    onRowsPerPageChange?.(parseInt(event.target.value, 10));
  };

  const currentSortBy = sortBy !== undefined ? sortBy : localSortBy;
  const currentSortOrder = sortBy !== undefined ? sortOrder : localSortOrder;

  return (
    <Paper sx={{ width: '100%', overflow: 'hidden' }}>
      <TableContainer>
        <Table stickyHeader>
          <TableHead>
            <TableRow>
              {selectable && (
                <TableCell padding="checkbox">
                  <Checkbox
                    color="primary"
                    indeterminate={selected.length > 0 && selected.length < rows.length}
                    checked={rows.length > 0 && selected.length === rows.length}
                    onChange={handleSelectAll}
                  />
                </TableCell>
              )}
              {columns.map((column) => (
                <TableCell
                  key={column.id}
                  align={column.align || 'left'}
                  style={{ width: column.width }}
                >
                  {column.sortable ? (
                    <TableSortLabel
                      active={currentSortBy === column.id}
                      direction={currentSortBy === column.id ? currentSortOrder : 'asc'}
                      onClick={() => handleSort(column.id)}
                    >
                      {column.label}
                    </TableSortLabel>
                  ) : (
                    column.label
                  )}
                </TableCell>
              ))}
            </TableRow>
          </TableHead>
          <TableBody>
            {rows.length === 0 ? (
              <TableRow>
                <TableCell
                  colSpan={columns.length + (selectable ? 1 : 0)}
                  align="center"
                  sx={{ py: 4 }}
                >
                  {emptyMessage}
                </TableCell>
              </TableRow>
            ) : (
              rows.map((row) => {
                const rowId = getRowId(row);
                const isItemSelected = isSelected(rowId);

                return (
                  <TableRow
                    hover
                    key={rowId}
                    selected={isItemSelected}
                    onClick={selectable ? () => handleSelectRow(rowId) : undefined}
                    sx={{ cursor: selectable ? 'pointer' : 'default' }}
                  >
                    {selectable && (
                      <TableCell padding="checkbox">
                        <Checkbox color="primary" checked={isItemSelected} />
                      </TableCell>
                    )}
                    {columns.map((column) => (
                      <TableCell key={column.id} align={column.align || 'left'}>
                        {column.render
                          ? column.render(row)
                          : (row as Record<string, unknown>)[column.id] as React.ReactNode}
                      </TableCell>
                    ))}
                  </TableRow>
                );
              })
            )}
          </TableBody>
        </Table>
      </TableContainer>
      {(onPageChange || onRowsPerPageChange) && (
        <TablePagination
          rowsPerPageOptions={rowsPerPageOptions}
          component="div"
          count={totalRows !== undefined ? totalRows : rows.length}
          rowsPerPage={rowsPerPage}
          page={page}
          onPageChange={handleChangePage}
          onRowsPerPageChange={handleChangeRowsPerPage}
          labelRowsPerPage="Строк на странице:"
          labelDisplayedRows={({ from, to, count }) =>
            `${from}-${to} из ${count !== -1 ? count : `более чем ${to}`}`
          }
        />
      )}
    </Paper>
  );
}

export default DataTable;
