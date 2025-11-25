import { useState, useEffect, useCallback, useRef } from 'react';
import { AxiosError } from 'axios';

interface UseApiState<T> {
  data: T | null;
  loading: boolean;
  error: string | null;
}

interface UseApiOptions {
  immediate?: boolean;
  deps?: unknown[];
}

export const useApi = <T>(
  apiFunc: () => Promise<T>,
  options: UseApiOptions = { immediate: true, deps: [] }
) => {
  const [state, setState] = useState<UseApiState<T>>({
    data: null,
    loading: options.immediate ?? true,
    error: null,
  });

  // Use ref to store the latest apiFunc to avoid infinite loops
  const apiFuncRef = useRef(apiFunc);
  apiFuncRef.current = apiFunc;

  const execute = useCallback(async () => {
    setState((prev) => ({ ...prev, loading: true, error: null }));

    try {
      const result = await apiFuncRef.current();
      setState({ data: result, loading: false, error: null });
      return result;
    } catch (err) {
      const error = err as AxiosError<{ message?: string; detail?: string }>;
      const errorMessage =
        error.response?.data?.message ||
        error.response?.data?.detail ||
        error.message ||
        'Произошла ошибка';
      setState((prev) => ({ ...prev, loading: false, error: errorMessage }));
      throw err;
    }
  }, []);

  useEffect(() => {
    if (options.immediate) {
      execute();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [options.immediate, ...(options.deps || [])]);

  const reset = useCallback(() => {
    setState({ data: null, loading: false, error: null });
  }, []);

  return {
    ...state,
    execute,
    refetch: execute,
    reset,
  };
};
