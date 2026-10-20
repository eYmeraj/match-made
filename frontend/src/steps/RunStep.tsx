import type { Dispatch } from 'react'
import type { JobStatus, MissingPolicy } from '../api/client'
import { CARDINALITIES } from '../options'
import { ErrorNotice, Notice, Spinner } from '../components/ui'
import type { Draft, DraftAction } from '../draft'
import { formatPercent } from '../format'

const MISSING: { value: MissingPolicy; label: string; help: string }[] = [
  {
    value: 'ignore',
    label: 'Ignore that pair',
    help: 'Score on the remaining column pairs, re-weighted.',
  },
  { value: 'mismatch', label: 'Count as a mismatch', help: 'That pair scores 0.' },
]

const STAGES: Record<string, string> = {
  load: 'Loading files',
  clean: 'Cleaning values',
  block: 'Finding candidate pairs',
  score: 'Scoring pairs',
}

export function RunStep({
  draft,
  dispatch,
  onStart,
  starting,
  startError,
}: {
  draft: Draft
  dispatch: Dispatch<DraftAction>
  onStart: () => void
  starting: boolean
  startError: unknown
}) {
  return (
    <div className="stack">
      <section className="card" aria-label="Threshold">
        <div className="card-head">
          <div className="titles">
            <h3>Match threshold</h3>
            <span className="subtle">
              Pairs scoring at or above this count as matches. You can adjust it on the results
              page without re-running.
            </span>
          </div>
          <span className="threshold-value num">{draft.threshold}</span>
        </div>
        <div className="card-body">
          <input
            className="range"
            type="range"
            min={0}
            max={100}
            step={1}
            aria-label="Match threshold"
            value={draft.threshold}
            onChange={(e) => dispatch({ type: 'setRun', patch: { threshold: Number(e.target.value) } })}
          />
        </div>
      </section>

      <section className="card" aria-label="Cardinality">
        <div className="card-head">
          <h3>How many matches per row</h3>
        </div>
        <div className="card-body">
          <div className="choice-cards">
            {CARDINALITIES.map((c) => (
              <label key={c.value} className="choice">
                <input
                  type="radio"
                  name="cardinality"
                  checked={draft.cardinality === c.value}
                  onChange={() => dispatch({ type: 'setRun', patch: { cardinality: c.value } })}
                />
                <span>
                  <strong>{c.label}</strong>
                  <span className="subtle">{c.help}</span>
                </span>
              </label>
            ))}
          </div>
        </div>
      </section>

      <section className="card" aria-label="Missing values">
        <div className="card-head">
          <h3>When a value is missing on either side</h3>
        </div>
        <div className="card-body">
          <div className="choice-cards">
            {MISSING.map((m) => (
              <label key={m.value} className="choice">
                <input
                  type="radio"
                  name="missing"
                  checked={draft.missing === m.value}
                  onChange={() => dispatch({ type: 'setRun', patch: { missing: m.value } })}
                />
                <span>
                  <strong>{m.label}</strong>
                  <span className="subtle">{m.help}</span>
                </span>
              </label>
            ))}
          </div>
        </div>
      </section>

      <ErrorNotice error={startError} />
      <div className="row" style={{ justifyContent: 'flex-end' }}>
        <button type="button" className="btn btn-primary" disabled={starting} onClick={onStart}>
          {starting ? <Spinner label="Starting…" /> : 'Run matching'}
        </button>
      </div>
    </div>
  )
}

export function JobProgress({
  job,
  onCancel,
  cancelling,
  onEdit,
}: {
  job: JobStatus
  onCancel: () => void
  cancelling: boolean
  onEdit: () => void
}) {
  const active = job.state === 'queued' || job.state === 'running'
  return (
    <section className="card" aria-label="Job progress">
      <div className="card-body stack">
        <div className="spread">
          {active ? (
            <Spinner
              label={job.state === 'queued' ? 'Waiting for a worker…' : (STAGES[job.stage ?? ''] ?? 'Working…')}
            />
          ) : (
            <h3>{job.state === 'cancelled' ? 'Run cancelled' : 'Run failed'}</h3>
          )}
          {active && <span className="num muted">{formatPercent(job.progress, 0)}</span>}
        </div>
        {active && (
          <div
            className="progress"
            role="progressbar"
            aria-valuemin={0}
            aria-valuemax={100}
            aria-valuenow={Math.round(job.progress * 100)}
          >
            <div style={{ width: `${job.progress * 100}%` }} />
          </div>
        )}
        {job.state === 'failed' && <Notice tone="error">{job.error ?? 'The run failed.'}</Notice>}
        <div className="row" style={{ justifyContent: 'flex-end' }}>
          {active ? (
            <button type="button" className="btn btn-danger" disabled={cancelling} onClick={onCancel}>
              Cancel run
            </button>
          ) : (
            <button type="button" className="btn" onClick={onEdit}>
              Edit setup
            </button>
          )}
        </div>
      </div>
    </section>
  )
}
