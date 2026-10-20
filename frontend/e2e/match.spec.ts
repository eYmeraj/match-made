import { expect, test, type Page } from '@playwright/test'
import { readFile } from 'node:fs/promises'

/** Upload the two fixtures and go to the Compare step; returns their dataset ids. */
async function uploadFiles(page: Page): Promise<string[]> {
  await page.goto('/')
  const ids: string[] = []
  for (const [file, card] of [
    ['a.csv', 'File A'],
    ['b.csv', 'File B'],
  ]) {
    const uploaded = page.waitForResponse((r) => r.url().endsWith('/api/datasets'))
    await page.getByTestId('file-input').first().setInputFiles(`e2e/fixtures/${file}`)
    ids.push((await (await uploaded).json()).id)
    await expect(page.getByRole('region', { name: card })).toContainText('2,000 rows')
  }
  await page.getByRole('button', { name: 'Continue to compare' }).click()
  return ids
}

async function addPair(page: Page, columnA: string, columnB: string, type: string) {
  const form = page.getByRole('region', { name: 'Add a column pair' })
  await form.getByLabel('Column in file A').selectOption(columnA)
  await form.getByLabel('Column in file B').selectOption(columnB)
  await form.getByLabel('Compare as').selectOption({ label: type })
  await form.getByRole('button', { name: 'Add column pair' }).click()
}

test('manual setup through to a downloaded merged file', async ({ page }) => {
  const datasetIds = await uploadFiles(page)

  // Manual configuration is the default: nothing is suggested until asked.
  await expect(page.getByText('No column pairs yet')).toBeVisible()
  await addPair(page, 'Company Name', 'vendor', 'Text')

  const comparison = page.getByRole('region', { name: 'Column pair Company Name and vendor' })
  await comparison.getByRole('button', { expanded: false }).click()
  await comparison.getByLabel('Remove legal suffixes').check()
  await expect(comparison.getByRole('cell', { name: 'jaskos first', exact: true })).toBeVisible()

  await page.getByRole('button', { name: 'Continue to blocking' }).click()
  await page.getByRole('button', { name: '+ Similar text rule' }).click()
  await page.getByRole('button', { name: 'Estimate', exact: true }).click()
  await expect(page.getByRole('region', { name: 'Estimate' })).toContainText('Pairs to score')

  await page.getByRole('button', { name: 'Continue to run' }).click()
  await page.getByRole('button', { name: 'Run matching' }).click()
  await expect(page.getByRole('heading', { name: 'Results' })).toBeVisible({ timeout: 60_000 })
  await expect(page).toHaveURL(/\?job=/)

  const matches = page.locator('.tile').filter({ hasText: 'Matches' }).locator('.value')
  await expect(matches).not.toHaveText('–')
  const count = Number((await matches.textContent())!.replace(/,/g, ''))
  expect(count).toBeGreaterThan(1000)
  await expect(page.locator('table.matches tbody tr')).toHaveCount(25)

  const downloading = page.waitForEvent('download')
  await page.getByRole('button', { name: 'Merged file' }).click()
  const download = await downloading
  expect(download.suggestedFilename()).toBe('merged.csv')
  const csv = await readFile((await download.path())!, 'utf8')
  expect(csv.split('\n')[0]).toMatch(/^match_score,/)
  expect(csv.trim().split('\n')).toHaveLength(count + 1)

  // The results page survives a reload from its URL.
  await page.reload()
  await expect(page.getByRole('heading', { name: 'Results' })).toBeVisible()
  await expect(matches).toHaveText(count.toLocaleString('en-US'))

  // Starting over keeps the uploads this run uses, so its link keeps working.
  const resultsUrl = page.url()
  const deletes = datasetIds.map((id) =>
    page.waitForResponse((r) => r.request().method() === 'DELETE' && r.url().endsWith(id)),
  )
  await page.getByRole('button', { name: 'Start over' }).click()
  for (const response of await Promise.all(deletes)) expect(response.status()).toBe(409)
  await page.goto(resultsUrl)
  await expect(matches).toHaveText(count.toLocaleString('en-US'))
  // The match table reads both uploaded files, so it only fills if they still exist.
  await expect(page.locator('table.matches tbody tr')).toHaveCount(25)
})

test('number fields take what is typed after deleting the old value', async ({ page }) => {
  await uploadFiles(page)
  await addPair(page, 'Company Name', 'vendor', 'Text')

  // Delete the default weight 1 and type 3, the way a person replaces a value.
  const weight = page.locator('.weight input')
  await weight.click()
  await weight.press('End')
  await weight.press('Backspace')
  await expect(weight).toHaveValue('')
  await weight.pressSequentially('3')
  await expect(weight).toHaveValue('3')
  await weight.blur()
  await expect(weight).toHaveValue('3')

  // A value outside the allowed range is not taken; leaving the field restores the last one.
  await weight.fill('500')
  await expect(weight).toHaveAttribute('aria-invalid', 'true')
  await weight.blur()
  await expect(weight).toHaveValue('3')
})

test('replacing a file deletes the unused upload', async ({ page }) => {
  await page.goto('/')
  await page.getByTestId('file-input').first().setInputFiles('e2e/fixtures/a.csv')
  const fileA = page.getByRole('region', { name: 'File A' })
  await expect(fileA).toContainText('2,000 rows')

  const deleted = page.waitForResponse((r) => r.request().method() === 'DELETE')
  await fileA.locator('input[type=file]').setInputFiles('e2e/fixtures/b.csv')
  await expect(fileA).toContainText('b.csv')
  const response = await deleted
  expect(response.status()).toBe(204)
  expect((await page.request.get(response.url())).status()).toBe(404)
})

test('auto-suggest proposes the matching column pairs', async ({ page }) => {
  await uploadFiles(page)
  await page.getByRole('button', { name: 'Auto-suggest' }).click()
  const heads = page.locator('.disclosure .pair')
  await expect(heads).toHaveText([
    'Company Name↔vendor',
    'Founded↔incorporated',
    'Revenue↔annual_revenue',
    'City↔city',
  ])
  await page.getByRole('button', { name: 'Continue to blocking' }).click()
  await expect(page.getByRole('heading', { name: /Rule 1 · Similar text/ })).toBeVisible()
})

test('a guessed encoding is shown with examples to check', async ({ page }) => {
  await page.goto('/')
  // Latin-1 bytes, as in a Windows "CSV" export; accented rows come late in the file.
  const rows = Array.from({ length: 20_000 }, (_, i) => `Plain Company ${i},Boston`)
  const csv = ['name,city', ...rows, 'Müller GmbH,München', 'São Paulo Ltda,São Paulo', ''].join('\n')
  await page.getByTestId('file-input').first().setInputFiles({
    name: 'windows.csv',
    mimeType: 'text/csv',
    buffer: Buffer.from(csv, 'latin1'),
  })
  const card = page.getByRole('region', { name: 'File A' })
  await expect(card).toContainText('20,002 rows')
  await expect(card.getByRole('combobox', { name: 'Encoding' })).toHaveValue('cp1252')
  await expect(card).toContainText('encoding was guessed as cp1252')
  await expect(card).toContainText('“Müller GmbH”')
})

test('match on any number of column pairs, added and removed freely', async ({ page }) => {
  await uploadFiles(page)
  await addPair(page, 'Company Name', 'vendor', 'Text')
  await addPair(page, 'Founded', 'incorporated', 'Date')
  await addPair(page, 'Revenue', 'annual_revenue', 'Number')
  await addPair(page, 'City', 'city', 'Category')
  const pairs = page.locator('.disclosure .pair')
  await expect(pairs).toHaveText([
    'Company Name↔vendor',
    'Founded↔incorporated',
    'Revenue↔annual_revenue',
    'City↔city',
  ])

  // After each pair the form offers a column not matched on yet; an exact repeat is refused.
  const form = page.getByRole('region', { name: 'Add a column pair' })
  await expect(form.getByLabel('Column in file A')).toHaveValue('company_id')
  await form.getByLabel('Column in file A').selectOption('Company Name')
  await form.getByLabel('Column in file B').selectOption('vendor')
  await expect(form.getByRole('button', { name: 'Add column pair' })).toBeDisabled()
  await expect(form).toContainText('This pair is already in the list.')

  await page.getByRole('button', { name: 'Remove pair City and city' }).click()
  await expect(pairs).toHaveText([
    'Company Name↔vendor',
    'Founded↔incorporated',
    'Revenue↔annual_revenue',
  ])

  await page.getByRole('button', { name: 'Continue to blocking' }).click()
  await page.getByRole('button', { name: '+ Similar text rule' }).click()
  await page.getByRole('button', { name: 'Continue to run' }).click()
  await page.getByRole('button', { name: 'Run matching' }).click()
  await expect(page.getByRole('heading', { name: 'Results' })).toBeVisible({ timeout: 60_000 })

  // Every remaining pair, and only those, is scored, shown and exported.
  await expect(page.locator('table.matches th.group')).toHaveText([
    'Company Name ↔ vendor',
    'Founded ↔ incorporated',
    'Revenue ↔ annual_revenue',
  ])
  const downloading = page.waitForEvent('download')
  await page.getByRole('button', { name: 'Merged file' }).click()
  const header = (await readFile((await (await downloading).path())!, 'utf8')).split('\n')[0]
  expect(header.split(',').filter((c) => c.startsWith('score: '))).toEqual([
    'score: Company Name ~ vendor',
    'score: Founded ~ incorporated',
    'score: Revenue ~ annual_revenue',
  ])
})
