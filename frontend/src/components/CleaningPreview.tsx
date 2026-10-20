import { useDeferredValue } from 'react'
import type { ColumnType } from '../api/client'
import { useCleaningPreview } from '../api/hooks'
import { sanitizeCleaning } from '../draft'
import { ErrorNotice, Spinner } from './ui'
import { formatPercent } from '../format'

export function CleaningPreview({
  title,
  datasetId,
  column,
  type,
  cleaning,
}: {
  title: string
  datasetId: string | null
  column: string
  type: ColumnType
  cleaning: Record<string, unknown>
}) {
  const deferred = useDeferredValue(sanitizeCleaning(cleaning))
  const preview = useCleaningPreview(datasetId, column, type, deferred)
  const data = preview.data
  const parses = type !== 'text' && type !== 'categorical'

  return (
    <div className="preview">
      <div className="spread">
        <span className="label">
          {title} · <span className="muted">{column}</span>
        </span>
        {preview.isFetching ? (
          <Spinner />
        ) : (
          data &&
          parses && (
            <span
              className={`badge ${data.parsed_fraction < 0.95 ? 'badge-warning' : 'badge-success'}`}
              title={`Of ${data.sampled_rows.toLocaleString()} non-empty sampled values`}
            >
              {formatPercent(data.parsed_fraction)} parsed
            </span>
          )
        )}
      </div>
      <ErrorNotice error={preview.error} />
      {data && (
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Original</th>
                <th>Cleaned</th>
              </tr>
            </thead>
            <tbody>
              {data.samples.length === 0 && (
                <tr>
                  <td colSpan={2} className="null">
                    No values
                  </td>
                </tr>
              )}
              {data.samples.map((s, i) => (
                <tr key={i}>
                  <td title={s.raw}>{s.raw}</td>
                  {s.cleaned === null ? (
                    <td className="null">{parses ? 'not parsed' : 'empty'}</td>
                  ) : (
                    <td title={s.cleaned}>{s.cleaned}</td>
                  )}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {data && data.failures.length > 0 && (
        <p className="subtle">
          Could not parse: {data.failures.map((f) => `“${f}”`).join(', ')}
        </p>
      )}
    </div>
  )
}
