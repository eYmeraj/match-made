import { keepPreviousData, useQuery } from '@tanstack/react-query'
import { ApiError, api, unwrap, type Cardinality, type ColumnType, type DatasetMeta } from './client'

const FINAL_STATES = new Set(['succeeded', 'failed', 'cancelled'])

export function useMeta() {
  return useQuery({
    queryKey: ['meta'],
    queryFn: () => unwrap(api.GET('/api/meta')),
    staleTime: Infinity,
  })
}

export function useDataset(id: string | null) {
  return useQuery({
    queryKey: ['dataset', id],
    queryFn: () => unwrap(api.GET('/api/datasets/{dataset_id}', { params: { path: { dataset_id: id! } } })),
    enabled: !!id,
    staleTime: Infinity,
    retry: (count, error) => !(error instanceof ApiError && error.status === 404) && count < 2,
  })
}

export function useRows(id: string | null, offset: number, limit: number) {
  return useQuery({
    queryKey: ['rows', id, offset, limit],
    queryFn: () =>
      unwrap(
        api.GET('/api/datasets/{dataset_id}/rows', {
          params: { path: { dataset_id: id! }, query: { offset, limit } },
        }),
      ),
    enabled: !!id,
    placeholderData: keepPreviousData,
    staleTime: Infinity,
  })
}

export function useCleaningPreview(
  datasetId: string | null,
  column: string,
  type: ColumnType,
  cleaning: unknown,
) {
  return useQuery({
    queryKey: ['preview', datasetId, column, type, cleaning],
    queryFn: () =>
      unwrap(
        api.POST('/api/preview/cleaning', {
          body: {
            dataset_id: datasetId!,
            column,
            type,
            cleaning: (cleaning ?? {}) as Record<string, never>,
          },
        }),
      ),
    enabled: !!datasetId && !!column,
    placeholderData: keepPreviousData,
    staleTime: Infinity,
  })
}

export function useJob(id: string | null) {
  return useQuery({
    queryKey: ['job', id],
    queryFn: () => unwrap(api.GET('/api/jobs/{job_id}', { params: { path: { job_id: id! } } })),
    enabled: !!id,
    refetchInterval: (query) =>
      query.state.data && FINAL_STATES.has(query.state.data.state) ? false : 1000,
  })
}

export function useJobConfig(id: string | null) {
  return useQuery({
    queryKey: ['job-config', id],
    queryFn: () =>
      unwrap(api.GET('/api/jobs/{job_id}/config', { params: { path: { job_id: id! } } })),
    enabled: !!id,
    staleTime: Infinity,
  })
}

export function useHistogram(id: string) {
  return useQuery({
    queryKey: ['histogram', id],
    queryFn: () =>
      unwrap(api.GET('/api/jobs/{job_id}/histogram', { params: { path: { job_id: id } } })),
    staleTime: Infinity,
  })
}

export function useStats(id: string, threshold: number, cardinality: Cardinality) {
  return useQuery({
    queryKey: ['stats', id, threshold, cardinality],
    queryFn: () =>
      unwrap(
        api.GET('/api/jobs/{job_id}/stats', {
          params: { path: { job_id: id }, query: { threshold, cardinality } },
        }),
      ),
    placeholderData: keepPreviousData,
    staleTime: Infinity,
  })
}

export function useMatches(
  id: string,
  threshold: number,
  cardinality: Cardinality,
  order: 'asc' | 'desc',
  offset: number,
  limit: number,
) {
  return useQuery({
    queryKey: ['matches', id, threshold, cardinality, order, offset, limit],
    queryFn: () =>
      unwrap(
        api.GET('/api/jobs/{job_id}/matches', {
          params: { path: { job_id: id }, query: { threshold, cardinality, order, offset, limit } },
        }),
      ),
    placeholderData: keepPreviousData,
    staleTime: Infinity,
  })
}

/** Upload a file with progress; multipart bodies are simplest with XHR. */
export function uploadDataset(
  file: File,
  onProgress: (fraction: number) => void,
  signal?: AbortSignal,
): Promise<DatasetMeta> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest()
    xhr.open('POST', '/api/datasets')
    xhr.responseType = 'json'
    xhr.upload.onprogress = (e) => e.lengthComputable && onProgress(e.loaded / e.total)
    xhr.upload.onload = () => onProgress(1)
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) resolve(xhr.response as DatasetMeta)
      else {
        const detail = (xhr.response as { detail?: unknown } | null)?.detail
        reject(new ApiError(xhr.status, typeof detail === 'string' ? detail : 'Upload failed'))
      }
    }
    xhr.onerror = () => reject(new ApiError(0, 'Could not reach the server'))
    signal?.addEventListener('abort', () => xhr.abort())
    const body = new FormData()
    body.append('file', file)
    xhr.send(body)
  })
}
