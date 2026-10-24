// Browser tests for the frontend. Starts the API with no models, and the tests mock
// /predict where they need specific predictions or errors.
const { defineConfig, devices } = require('@playwright/test');

const PORT = 8765;

module.exports = defineConfig({
  testDir: 'tests/e2e',
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  reporter: process.env.CI ? 'github' : 'list',
  use: {
    baseURL: `http://127.0.0.1:${PORT}`,
    trace: 'retain-on-failure',
  },
  projects: [{ name: 'chromium', use: { ...devices['Desktop Chrome'] } }],
  webServer: {
    command: 'python -m api.main',
    url: `http://127.0.0.1:${PORT}/health`,
    env: { PORT: String(PORT), MODELS_DIR: 'tests/e2e/no-models' },
    reuseExistingServer: false,
    timeout: 120_000,
  },
});
