import { defineConfig, devices } from '@playwright/test'

// Runs against a live stack: `docker compose up -d --build` from the repo root.
export default defineConfig({
  testDir: './e2e',
  timeout: 120_000,
  fullyParallel: true,
  reporter: 'list',
  use: {
    baseURL: process.env.E2E_BASE_URL ?? 'http://localhost:8080',
    trace: 'retain-on-failure',
  },
  projects: [{ name: 'chromium', use: { ...devices['Desktop Chrome'] } }],
})
