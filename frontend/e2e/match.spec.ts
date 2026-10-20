import { expect, test, type Page } from '@playwright/test'
import { readFile } from 'node:fs/promises'

async function uploadFiles(page: Page) {
  await page.goto('/')
  await page.getByTestId('file-input').first().setInputFiles('e2e/fixtures/a.csv')
  await expect(page.getByRole('region', { name: 'File A' })).toContainText('2,000 rows')
  await page.getByTestId('file-input').first().setInputFiles('e2e/fixtures/b.csv')
  await expect(page.getByRole('region', { name: 'File B' })).toContainText('2,000 rows')
  await page.getByRole('button', { name: 'Continue to compare' }).click()
}

test('manual setup through to a downloaded merged file', async ({ page }) => {
  await uploadFiles(page)

  // Manual configuration is the default: nothing is suggested until asked.
  await expect(page.getByText('No comparisons yet')).toBeVisible()
  await page.getByLabel('Column in file A').selectOption('Company Name')
  await page.getByLabel('Column in file B').selectOption('vendor')
  await page.getByRole('button', { name: 'Add comparison' }).click()

  const comparison = page.getByRole('region', { name: 'Comparison Company Name and vendor' })
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
