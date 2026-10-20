import createClient from 'openapi-fetch'
import type { components, paths } from './schema'

export type Schemas = components['schemas']
export type DatasetMeta = Schemas['DatasetMeta']
export type ColumnProfile = Schemas['ColumnProfile']
export type ColumnType = Schemas['ColumnType']
export type Comparison = Schemas['MatchConfig']['comparisons'][number]
export type BlockingRule = NonNullable<Schemas['Blocking']['rules']>[number]
export type KeyRule = Schemas['ExactBlockingRule']
export type TfidfRule = Schemas['TfidfBlockingRule']
export type KeyPart = Schemas['BlockingKeyPart']
export type KeyTransform = Schemas['KeyTransform']
export type MatchConfig = Schemas['MatchConfig']
export type Cardinality = Schemas['Cardinality']
export type MissingPolicy = Schemas['MissingPolicy']
export type JobStatus = Schemas['JobStatus']
export type Meta = Schemas['Meta']
export type ExportKind = Schemas['ExportKind']
export type JoinMode = Schemas['JoinMode']

export const api = createClient<paths>({ baseUrl: '' })

export class ApiError extends Error {
  readonly status: number

  constructor(status: number, message: string) {
    super(message)
    this.status = status
  }
}

function describe(error: unknown): string {
  if (error && typeof error === 'object' && 'detail' in error) {
    const { detail } = error as { detail: unknown }
    if (typeof detail === 'string') return detail
    if (Array.isArray(detail)) {
      return detail
        .map((d: { loc?: unknown[]; msg?: string }) => {
          const where = (d.loc ?? []).filter((p) => p !== 'body').join('.')
          return where ? `${where}: ${d.msg}` : (d.msg ?? '')
        })
        .join('; ')
    }
  }
  return 'Unexpected server error'
}

/** Return the response data or throw an ApiError with the server's message. */
export async function unwrap<T>(
  call: Promise<{ data?: T; error?: unknown; response: Response }>,
): Promise<T> {
  const { data, error, response } = await call
  if (!response.ok) throw new ApiError(response.status, describe(error))
  return data as T
}

export function exportUrl(
  jobId: string,
  params: {
    kind: ExportKind
    threshold: number
    cardinality: Cardinality
    format: 'csv' | 'xlsx'
    join?: JoinMode
  },
): string {
  const query = new URLSearchParams({
    kind: params.kind,
    threshold: String(params.threshold),
    cardinality: params.cardinality,
    format: params.format,
    ...(params.join ? { join: params.join } : {}),
  })
  return `/api/jobs/${jobId}/export?${query}`
}
