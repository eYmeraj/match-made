import { useMutation, useQueryClient } from '@tanstack/react-query'
import { useEffect, useReducer, useState } from 'react'
import { ApiError, api, discardDataset, unwrap, type Meta } from './api/client'
import { useDataset, useJob, useJobConfig, useMeta } from './api/hooks'
import { ErrorNotice, Spinner } from './components/ui'
import { draftReducer, emptyDraft, toConfig, type Draft, type DraftAction } from './draft'
import { BlockingStep } from './steps/BlockingStep'
import { ComparisonsStep } from './steps/ComparisonsStep'
import { FilesStep } from './steps/FilesStep'
import { ResultsStep } from './steps/ResultsStep'
import { JobProgress, RunStep } from './steps/RunStep'

type Step = 'files' | 'compare' | 'blocking' | 'run' | 'results'

const STEPS: { id: Step; label: string; title: string; intro: string }[] = [
  { id: 'files', label: 'Files', title: 'Upload two files', intro: 'Rows in file A are matched against rows in file B.' },
  { id: 'compare', label: 'Compare', title: 'Choose the columns to match on', intro: '' },
  { id: 'blocking', label: 'Blocking', title: 'Limit which pairs are scored', intro: '' },
  { id: 'run', label: 'Run', title: 'Run matching', intro: 'Set how strict matching is and how many matches each row may have.' },
  { id: 'results', label: 'Results', title: 'Results', intro: 'Tune the threshold, check borderline matches and download the files.' },
]

const DRAFT_KEY = 'matchmade.draft'

function loadDraft(): Draft {
  try {
    const saved = sessionStorage.getItem(DRAFT_KEY)
    return saved ? { ...emptyDraft, ...JSON.parse(saved) } : emptyDraft
  } catch {
    return emptyDraft
  }
}

function jobFromUrl(): string | null {
  return new URLSearchParams(window.location.search).get('job')
}

function setJobInUrl(id: string | null) {
  const url = new URL(window.location.href)
  if (id) url.searchParams.set('job', id)
  else url.searchParams.delete('job')
  window.history.replaceState(null, '', url)
}

export default function App() {
  const meta = useMeta()
  if (meta.isPending) {
    return (
      <div className="page">
        <Spinner label="Connecting…" />
      </div>
    )
  }
  if (meta.isError) {
    return (
      <div className="page">
        <ErrorNotice error={new Error(`Cannot reach the Match Made server. ${meta.error.message}`)} />
      </div>
    )
  }
  return <Workspace meta={meta.data} />
}

const isNotFound = (error: unknown) => error instanceof ApiError && error.status === 404

function Workspace({ meta }: { meta: Meta }) {
  const [draft, dispatch] = useReducer(
    (state: Draft, action: DraftAction) => draftReducer(state, action, meta.transforms),
    undefined,
    loadDraft,
  )
  const [jobId, setJobId] = useState<string | null>(jobFromUrl)
  const [chosenStep, setStep] = useState<Step>(jobId ? 'run' : 'files')
  const queryClient = useQueryClient()

  const datasetA = useDataset(draft.datasetA)
  const datasetB = useDataset(draft.datasetB)
  const job = useJob(jobId)
  const jobConfig = useJobConfig(jobId)

  useEffect(() => {
    try {
      sessionStorage.setItem(DRAFT_KEY, JSON.stringify(draft))
    } catch {
      // Storage can be unavailable (private mode); the draft just won't survive a reload.
    }
  }, [draft])

  // Opening a job link restores the setup that produced it.
  useEffect(() => {
    if (jobConfig.data) dispatch({ type: 'load', config: jobConfig.data })
  }, [jobConfig.data])

  // A dataset deleted on the server (or a stale session) is dropped from the draft. Other
  // errors, such as the API restarting, are retried rather than losing the upload.
  const goneA = isNotFound(datasetA.error)
  const goneB = isNotFound(datasetB.error)
  useEffect(() => {
    if (goneA) dispatch({ type: 'setDataset', side: 'a', dataset: null })
    if (goneB) dispatch({ type: 'setDataset', side: 'b', dataset: null })
  }, [goneA, goneB])

  const jobState = job.data?.state
  // A finished run shows its results; the run page only tracks progress.
  const step: Step = chosenStep === 'run' && jobState === 'succeeded' ? 'results' : chosenStep

  // On narrow screens the stepper scrolls; keep the current step visible.
  useEffect(() => {
    document.querySelector('.stepper .current')?.scrollIntoView({ block: 'nearest', inline: 'nearest' })
  }, [step])

  const selectJob = (id: string | null) => {
    setJobId(id)
    setJobInUrl(id)
  }

  const start = useMutation({
    mutationFn: () => unwrap(api.POST('/api/jobs', { body: toConfig(draft)! })),
    onSuccess: (status) => {
      queryClient.setQueryData(['job', status.id], status)
      selectJob(status.id)
    },
  })

  const cancel = useMutation({
    mutationFn: (id: string) =>
      unwrap(api.POST('/api/jobs/{job_id}/cancel', { params: { path: { job_id: id } } })),
    onSuccess: (status) => queryClient.setQueryData(['job', status.id], status),
  })

  const editSetup = () => {
    selectJob(null)
    start.reset()
    setStep('compare')
  }

  const startOver = () => {
    // A run that has not finished would otherwise keep the only worker busy, unseen.
    if (job.data?.state === 'queued' || job.data?.state === 'running') cancel.mutate(job.data.id)
    for (const id of [draft.datasetA, draft.datasetB]) if (id) discardDataset(id)
    selectJob(null)
    dispatch({ type: 'reset' })
    setStep('files')
  }

  const haveFiles = !!datasetA.data && !!datasetB.data
  const available: Record<Step, boolean> = {
    files: true,
    compare: haveFiles,
    blocking: haveFiles && draft.comparisons.length > 0,
    run: haveFiles && draft.comparisons.length > 0,
    results: jobState === 'succeeded',
  }
  const order = STEPS.map((s) => s.id)
  const index = order.indexOf(step)
  const current = STEPS[index]
  const next = order[index + 1] as Step | undefined
  // While a job exists the setup is frozen; editing starts a new run.
  const locked = !!jobId && step !== 'results'

  return (
    <>
      <header className="app-header">
        <div className="app-header-inner">
          <div className="brand">
            <img src="/favicon.svg" alt="" />
            Match Made
          </div>
          <nav aria-label="Steps">
            <ol className="stepper">
              {STEPS.map((s, i) => (
                <li key={s.id} className={s.id === step ? 'current' : i < index ? 'done' : undefined}>
                  <button
                    type="button"
                    disabled={!available[s.id] || (locked && s.id !== 'run')}
                    aria-current={s.id === step ? 'step' : undefined}
                    onClick={() => setStep(s.id)}
                  >
                    <span className="dot">{i + 1}</span>
                    {s.label}
                  </button>
                </li>
              ))}
            </ol>
          </nav>
        </div>
      </header>

      <main className="page">
        <div className="page-head">
          <div>
            <h1>{current.title}</h1>
            {current.intro && <p className="muted">{current.intro}</p>}
          </div>
          {(step === 'results' || (step !== 'files' && (draft.datasetA || draft.datasetB))) && (
            <div className="row">
              {step === 'results' && (
                <button type="button" className="btn" onClick={editSetup}>
                  Edit setup
                </button>
              )}
              <button type="button" className="btn btn-ghost" onClick={startOver}>
                Start over
              </button>
            </div>
          )}
        </div>

        {step === 'files' && (
          <FilesStep
            datasetA={draft.datasetA}
            datasetB={draft.datasetB}
            onChange={(side, dataset) => {
              // A replaced, re-read or removed upload is no longer needed.
              const previous = side === 'a' ? draft.datasetA : draft.datasetB
              if (previous && previous !== dataset?.id) discardDataset(previous)
              dispatch({ type: 'setDataset', side, dataset })
            }}
          />
        )}
        {step === 'compare' && haveFiles && (
          <ComparisonsStep
            draft={draft}
            dispatch={dispatch}
            meta={meta}
            datasetA={datasetA.data}
            datasetB={datasetB.data}
          />
        )}
        {step === 'blocking' && haveFiles && (
          <BlockingStep
            draft={draft}
            dispatch={dispatch}
            transforms={meta.transforms} rowsA={datasetA.data.rows} rowsB={datasetB.data.rows} />
        )}
        {step === 'run' &&
          (jobId ? (
            job.data ? (
              <JobProgress
                job={job.data}
                cancelling={cancel.isPending}
                onCancel={() => cancel.mutate(job.data.id)}
                onEdit={editSetup}
              />
            ) : job.isError ? (
              <>
                <ErrorNotice error={job.error} />
                <div>
                  <button type="button" className="btn" onClick={startOver}>
                    Start over
                  </button>
                </div>
              </>
            ) : (
              <Spinner label="Loading…" />
            )
          ) : (
            <RunStep
              draft={draft}
              dispatch={dispatch}
              onStart={() => start.mutate()}
              starting={start.isPending}
              startError={start.error}
            />
          ))}
        {step === 'results' && job.data?.state === 'succeeded' && job.data.summary && (
          <ResultsStep
            key={job.data.id}
            job={job.data}
            comparisons={jobConfig.data?.comparisons ?? draft.comparisons}
            initialThreshold={jobConfig.data?.threshold ?? draft.threshold}
            initialCardinality={jobConfig.data?.cardinality ?? draft.cardinality}
          />
        )}

        {!locked && next && next !== 'results' && step !== 'run' && (
          <div className="footer-nav">
            <span />
            <button
              type="button"
              className="btn btn-primary"
              disabled={!available[next]}
              onClick={() => setStep(next)}
            >
              Continue to {STEPS[index + 1].label.toLowerCase()}
            </button>
          </div>
        )}
      </main>
    </>
  )
}
