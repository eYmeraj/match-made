import { describe, expect, it } from 'vitest'
import type { BlockingRule, Comparison, DatasetMeta, Meta } from './api/client'
import {
  comparisonId,
  draftReducer,
  emptyDraft,
  makeComparison,
  reconcileRules,
  toConfig,
} from './draft'

const meta = {
  transforms: {},
  metrics: [],
  default_cleaning: { text: { lowercase: true }, date: { day_first: false } },
  keep_margin: 25,
  max_candidates: 100,
} as unknown as Meta

const name = makeComparison(meta, 'text', 'Name', 'vendor', 'name')
const founded = makeComparison(meta, 'date', 'Founded', 'incorporated', 'founded')

describe('comparisonId', () => {
  it('slugs the column and avoids taken ids', () => {
    expect(comparisonId('Company Name', [])).toBe('company_name')
    expect(comparisonId('Company Name', ['company_name', 'company_name_2'])).toBe('company_name_3')
    expect(comparisonId('???', [])).toBe('field')
  })
})

describe('reconcileRules', () => {
  it('drops parts and rules that reference removed comparisons', () => {
    const rules: BlockingRule[] = [
      {
        kind: 'key',
        parts: [
          { comparison_id: 'name', transform: 'first_token', param: null },
          { comparison_id: 'gone', transform: 'exact', param: null },
        ],
      },
      { kind: 'key', parts: [{ comparison_id: 'gone', transform: 'exact', param: null }] },
      { kind: 'tfidf', comparison_id: 'gone', top_k: 10, min_similarity: 0.3 },
    ]
    expect(reconcileRules(rules, [name])).toEqual([
      { kind: 'key', parts: [{ comparison_id: 'name', transform: 'first_token', param: null }] },
    ])
  })

  it('falls back to exact keys and drops text rules when a type changes', () => {
    const asDate = { ...founded, id: 'name' } as Comparison
    const rules: BlockingRule[] = [
      { kind: 'key', parts: [{ comparison_id: 'name', transform: 'prefix', param: 3 }] },
      { kind: 'tfidf', comparison_id: 'name', top_k: 10, min_similarity: 0.3 },
    ]
    expect(reconcileRules(rules, [asDate])).toEqual([
      { kind: 'key', parts: [{ comparison_id: 'name', transform: 'exact', param: null }] },
    ])
  })
})

describe('draftReducer', () => {
  it('keeps comparisons whose columns exist in a replacement file', () => {
    const draft = { ...emptyDraft, comparisons: [name, founded] }
    const dataset = { id: 'b2', columns: [{ name: 'vendor' }] } as DatasetMeta
    const next = draftReducer(draft, { type: 'setDataset', side: 'b', dataset })
    expect(next.datasetB).toBe('b2')
    expect(next.comparisons.map((c) => c.id)).toEqual(['name'])
  })
})

describe('toConfig', () => {
  it('needs both files and a comparison, and drops unfinished replacements', () => {
    expect(toConfig(emptyDraft)).toBeNull()
    const withReplacements = {
      ...name,
      cleaning: {
        ...name.cleaning,
        replacements: [
          { find: '', replace: 'x', regex: false },
          { find: '&', replace: 'and', regex: false },
        ],
      },
    } as Comparison
    const config = toConfig({
      ...emptyDraft,
      datasetA: 'a',
      datasetB: 'b',
      comparisons: [withReplacements],
    })
    expect(config?.comparisons[0].cleaning).toMatchObject({
      replacements: [{ find: '&', replace: 'and', regex: false }],
    })
  })
})
