import { useState, useMemo } from 'react';

interface UsePaginationProps {
  totalItems: number;
  itemsPerPage: number;
  initialPage?: number;
}

export const usePagination = ({
  totalItems,
  itemsPerPage,
  initialPage = 0,
}: UsePaginationProps) => {
  const [currentPage, setCurrentPage] = useState(initialPage);

  const totalPages = useMemo(() => Math.ceil(totalItems / itemsPerPage), [totalItems, itemsPerPage]);

  const offset = useMemo(() => currentPage * itemsPerPage, [currentPage, itemsPerPage]);

  const goToPage = (page: number) => {
    const pageNumber = Math.max(0, Math.min(page, totalPages - 1));
    setCurrentPage(pageNumber);
  };

  const goToNextPage = () => {
    goToPage(currentPage + 1);
  };

  const goToPrevPage = () => {
    goToPage(currentPage - 1);
  };

  const goToFirstPage = () => {
    goToPage(0);
  };

  const goToLastPage = () => {
    goToPage(totalPages - 1);
  };

  const reset = () => {
    setCurrentPage(initialPage);
  };

  return {
    currentPage,
    totalPages,
    offset,
    limit: itemsPerPage,
    goToPage,
    goToNextPage,
    goToPrevPage,
    goToFirstPage,
    goToLastPage,
    reset,
    hasNext: currentPage < totalPages - 1,
    hasPrev: currentPage > 0,
  };
};
