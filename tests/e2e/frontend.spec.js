// @ts-check
const { test, expect } = require('@playwright/test');

const prediction = (label, confidence) => ({ label, confidence });

async function fillForm(page) {
  await page.getByLabel('Headline').fill('Senate passes budget');
  await page.getByLabel('Article text').fill('The Senate voted on Tuesday to pass the budget bill.');
}

async function mockPredict(page, status, body, contentType = 'application/json') {
  await page.route('**/predict', (route) =>
    route.fulfill({ status, contentType, body: typeof body === 'string' ? body : JSON.stringify(body) }),
  );
}

// Fails the test on any console error (including Content-Security-Policy violations),
// uncaught exception or alert() dialog.
let problems;

test.beforeEach(async ({ page }) => {
  problems = [];
  page.on('console', (msg) => {
    // Chrome logs every non-2xx response like this; the error tests trigger those on purpose.
    if (msg.type() === 'error' && !msg.text().startsWith('Failed to load resource')) {
      problems.push(`console error: ${msg.text()}`);
    }
  });
  page.on('pageerror', (error) => problems.push(`uncaught: ${error.message}`));
  page.on('dialog', async (dialog) => {
    problems.push(`unexpected dialog: ${dialog.message()}`);
    await dialog.dismiss();
  });
});

test.afterEach(() => {
  expect(problems).toEqual([]);
});

test('real API without models reports both models unavailable', async ({ page }) => {
  await page.goto('/');
  await fillForm(page);
  await page.getByRole('button', { name: 'Analyze article' }).click();

  await expect(page.locator('#summary')).toHaveText(/No models are loaded/);
  await expect(page.locator('#ml-label')).toHaveText('Unavailable');
  await expect(page.locator('#ai-meter')).toBeHidden();
});

test.describe('submitting', () => {
  test('Enter in the title and Ctrl+Enter in the article submit the form', async ({ page }) => {
    let calls = 0;
    await page.route('**/predict', (route) => {
      calls++;
      route.fulfill({ json: { ml: prediction('REAL', 0.9), ai: prediction('REAL', 0.8) } });
    });
    await page.goto('/');
    await fillForm(page);

    await page.getByLabel('Headline').press('Enter');
    await expect(page.locator('#results-panel')).toBeVisible();
    expect(calls).toBe(1);

    await page.getByLabel('Article text').press('Control+Enter');
    await expect.poll(() => calls).toBe(2);
  });

  test('shows loading state and blocks double submits', async ({ page }) => {
    let calls = 0;
    let release;
    await page.route('**/predict', async (route) => {
      calls++;
      await new Promise((resolve) => (release = resolve));
      await route.fulfill({ json: { ml: prediction('REAL', 0.9), ai: prediction('REAL', 0.8) } });
    });
    await page.goto('/');
    await fillForm(page);

    const button = page.getByRole('button', { name: 'Analyze article' });
    await button.click();
    await expect(button).toBeDisabled();
    await expect(button).toHaveAttribute('aria-busy', 'true');
    await expect(page.locator('#loader')).toBeVisible();
    await expect(page.getByRole('status')).toHaveText(/Analyzing/);
    await page.getByLabel('Headline').press('Enter');

    release();
    await expect(page.locator('#results-panel')).toBeVisible();
    await expect(button).toBeEnabled();
    await expect(page.locator('#loader')).toBeHidden();
    await expect(page.getByRole('status')).toHaveText(/Results ready\. Both models predict REAL\./);
    expect(calls).toBe(1);
  });
});

test.describe('validation', () => {
  test('missing fields show inline errors without sending a request', async ({ page }) => {
    let calls = 0;
    await page.route('**/predict', (route) => {
      calls++;
      route.abort();
    });
    await page.goto('/');
    await page.getByLabel('Article text').fill('Body only');
    await page.getByLabel('Headline').fill('   ');
    await page.getByRole('button', { name: 'Analyze article' }).click();

    const title = page.getByLabel('Headline');
    await expect(page.locator('#title-error')).toHaveText('Please enter the headline.');
    await expect(title).toHaveAttribute('aria-invalid', 'true');
    await expect(title).toBeFocused();
    await expect(title).toHaveCSS('border-bottom-color', 'rgb(209, 32, 26)');
    expect(calls).toBe(0);

    await title.fill('A title');
    await expect(page.locator('#title-error')).toBeEmpty();
    await expect(title).not.toHaveAttribute('aria-invalid');
  });

  test('inputs are limited to what the API accepts, with a character counter', async ({ page }) => {
    await page.goto('/');
    await expect(page.getByLabel('Headline')).toHaveAttribute('maxlength', '1000');
    await expect(page.getByLabel('Article text')).toHaveAttribute('maxlength', '100000');

    await page.getByLabel('Headline').fill('x'.repeat(950));
    await expect(page.locator('#title-count')).toHaveText('950 / 1,000');
    await expect(page.locator('#title-count')).toHaveClass(/near-limit/);
  });
});

test.describe('errors are shown inline', () => {
  const cases = [
    ['500 with a plain-text body', 500, 'Internal Server Error', 'text/plain', /error \(500\)/],
    ['502 HTML page from a proxy', 502, '<html>Bad gateway</html>', 'text/html', /temporarily unavailable/],
    ['413 body too large', 413, { detail: 'Request body must be at most 1000000 bytes' }, undefined, /too long/],
    [
      '422 validation error',
      422,
      { detail: [{ loc: ['body', 'text'], msg: 'String should have at most 100000 characters' }] },
      undefined,
      /Please check your input: text: String should have at most 100000 characters/,
    ],
    ['429 rate limited', 429, { detail: 'Too Many Requests' }, undefined, /Too many requests/],
    ['200 with an unexpected body', 200, { unexpected: true }, undefined, /unexpected response/],
  ];

  for (const [name, status, body, contentType, message] of cases) {
    test(name, async ({ page }) => {
      await mockPredict(page, status, body, contentType);
      await page.goto('/');
      await fillForm(page);
      await page.getByRole('button', { name: 'Analyze article' }).click();

      await expect(page.getByRole('alert')).toHaveText(message);
      await expect(page.getByRole('button', { name: 'Analyze article' })).toBeEnabled();
      await expect(page.locator('#results-panel')).toBeHidden();
    });
  }

  test('network failure', async ({ page }) => {
    await page.route('**/predict', (route) => route.abort('failed'));
    await page.goto('/');
    await fillForm(page);
    await page.getByRole('button', { name: 'Analyze article' }).click();
    await expect(page.getByRole('alert')).toHaveText(/Could not reach the API/);
  });

  test('a request that never answers times out after 60 seconds', async ({ page }) => {
    await page.clock.install();
    await page.route('**/predict', () => {});
    await page.goto('/');
    await fillForm(page);
    await page.getByRole('button', { name: 'Analyze article' }).click();
    await expect(page.locator('#loader')).toBeVisible();

    await page.clock.runFor(61_000);
    await expect(page.getByRole('alert')).toHaveText(/timed out/);
    await expect(page.getByRole('button', { name: 'Analyze article' })).toBeEnabled();
    await expect(page.locator('#loader')).toBeHidden();
  });

  test('the error is cleared by the next successful request', async ({ page }) => {
    await mockPredict(page, 500, 'boom', 'text/plain');
    await page.goto('/');
    await fillForm(page);
    await page.getByRole('button', { name: 'Analyze article' }).click();
    await expect(page.getByRole('alert')).toBeVisible();

    await page.unrouteAll();
    await mockPredict(page, 200, { ml: prediction('FAKE', 0.9), ai: prediction('FAKE', 0.95) });
    await page.getByRole('button', { name: 'Analyze article' }).click();
    await expect(page.locator('#summary')).toHaveText('Both models predict FAKE.');
    await expect(page.locator('#error-banner')).toBeHidden();
  });
});

test.describe('results', () => {
  test('disagreement, low confidence and confidence bars', async ({ page }) => {
    await mockPredict(page, 200, { ml: prediction('REAL', 0.69), ai: prediction('FAKE', 0.55) });
    await page.goto('/');
    await fillForm(page);
    await page.getByRole('button', { name: 'Analyze article' }).click();

    await expect(page.locator('#summary')).toHaveText(/models disagree/);
    await expect(page.locator('#ml-label')).toHaveText('REAL');
    await expect(page.locator('#ml-label')).toHaveClass(/real/);
    await expect(page.locator('#ai-label')).toHaveClass(/fake/);
    await expect(page.locator('#ml-conf')).toHaveText('69.0%');
    await expect(page.locator('#ml-meter')).toHaveAttribute('aria-valuenow', '69');
    await expect(page.locator('#ml-low')).toBeHidden();
    await expect(page.locator('#ai-low')).toBeVisible();

    await expect
      .poll(() =>
        page.evaluate(() => {
          const bar = document.getElementById('ml-bar').getBoundingClientRect().width;
          return bar / document.getElementById('ml-meter').getBoundingClientRect().width;
        }),
      )
      .toBeCloseTo(0.69, 2);
  });

  test('only one model available', async ({ page }) => {
    await mockPredict(page, 200, { ml: prediction('REAL', 0.9), ai: prediction('UNAVAILABLE', null) });
    await page.goto('/');
    await fillForm(page);
    await page.getByRole('button', { name: 'Analyze article' }).click();

    await expect(page.locator('#summary')).toHaveText('Only one model is available. It predicts REAL.');
    await expect(page.locator('#ai-label')).toHaveText('Unavailable');
    await expect(page.locator('#ai-conf')).toHaveText('Model not loaded on the server');
    await expect(page.locator('#ai-meter')).toBeHidden();
    await expect(page.locator('#ai-low')).toBeHidden();
  });

  test('confidence bars animate again on a repeat result', async ({ page }) => {
    await mockPredict(page, 200, { ml: prediction('REAL', 0.9), ai: prediction('REAL', 0.9) });
    await page.goto('/');
    await fillForm(page);
    const button = page.getByRole('button', { name: 'Analyze article' });
    await button.click();
    await expect(page.locator('#results-panel')).toBeVisible();
    await page.waitForTimeout(1200);

    await button.click();
    await expect(page.locator('#results-panel')).toBeVisible();
    await page.waitForTimeout(100);
    const ratio = await page.evaluate(
      () =>
        document.getElementById('ml-bar').getBoundingClientRect().width /
        document.getElementById('ml-meter').getBoundingClientRect().width,
    );
    expect(ratio).toBeGreaterThan(0);
    expect(ratio).toBeLessThan(0.85);
  });
});

test.describe('3D newspaper stack', () => {
  const stampCases = [
    ['both REAL', { ml: prediction('REAL', 0.9), ai: prediction('REAL', 0.8) }, 'real', 'Real'],
    ['both FAKE', { ml: prediction('FAKE', 0.9), ai: prediction('FAKE', 0.7) }, 'fake', 'Fake'],
    ['models disagree', { ml: prediction('REAL', 0.9), ai: prediction('FAKE', 0.8) }, 'mixed', 'Split'],
    ['agree but both unsure', { ml: prediction('REAL', 0.55), ai: prediction('REAL', 0.52) }, 'mixed', 'Unsure'],
    ['one model only', { ml: prediction('FAKE', 0.9), ai: prediction('UNAVAILABLE', null) }, 'fake', 'Fake'],
    ['no models', { ml: prediction('UNAVAILABLE', null), ai: prediction('UNAVAILABLE', null) }, 'none', 'No model'],
  ];

  for (const [name, body, state, stamp] of stampCases) {
    test(`stamps the stack: ${name}`, async ({ page }) => {
      await mockPredict(page, 200, body);
      await page.goto('/');
      await fillForm(page);
      await page.getByRole('button', { name: 'Analyze article' }).click();

      await expect(page.locator('#scene')).toHaveAttribute('data-state', state);
      await expect(page.locator('#stamp-text')).toHaveText(stamp);
      await expect(page.locator('#method-number')).toHaveText('03');
    });
  }

  test('fans out while analyzing and resets after an error', async ({ page }) => {
    let release;
    await page.route('**/predict', async (route) => {
      await new Promise((resolve) => (release = resolve));
      await route.fulfill({ status: 500, body: 'boom' });
    });
    await page.goto('/');
    await expect(page.locator('#scene')).toHaveAttribute('data-state', 'idle');
    await fillForm(page);
    await page.getByRole('button', { name: 'Analyze article' }).click();

    await expect(page.locator('#scene')).toHaveAttribute('data-state', 'analyzing');
    release();
    await expect(page.getByRole('alert')).toBeVisible();
    await expect(page.locator('#scene')).toHaveAttribute('data-state', 'idle');
    await expect(page.locator('#stamp-text')).toBeEmpty();
  });

  test('tilts towards the pointer', async ({ page }) => {
    await page.goto('/');
    await page.mouse.move(10, 10);
    await expect.poll(() => page.locator('#scene').evaluate((el) => el.style.getPropertyValue('--rz'))).toMatch(/^-/);
    await page.mouse.move(1200, 700);
    await expect.poll(() => page.locator('#scene').evaluate((el) => el.style.getPropertyValue('--rz'))).toMatch(/^\d/);
  });

  test('is hidden from assistive technology', async ({ page }) => {
    await page.goto('/');
    await expect(page.locator('#scene')).toHaveAttribute('aria-hidden', 'true');
  });
});

test.describe('mobile layout', () => {
  for (const width of [320, 375, 414]) {
    test(`no horizontal scrolling while analyzing at ${width}px`, async ({ page }) => {
      await page.setViewportSize({ width, height: 800 });
      await page.route('**/predict', () => {});
      await page.goto('/');
      await fillForm(page);
      await page.getByRole('button', { name: 'Analyze article' }).click();
      await expect(page.locator('#scene')).toHaveAttribute('data-state', 'analyzing');
      await page.waitForTimeout(900);

      const [scrollWidth, innerWidth] = await page.evaluate(() => [document.documentElement.scrollWidth, innerWidth]);
      expect(scrollWidth).toBeLessThanOrEqual(innerWidth);
    });

    test(`no horizontal scrolling at ${width}px`, async ({ page }) => {
      await page.setViewportSize({ width, height: 800 });
      await mockPredict(page, 200, { ml: prediction('REAL', 0.69), ai: prediction('REAL', 0.5) });
      await page.goto('/');
      await fillForm(page);
      await page.getByRole('button', { name: 'Analyze article' }).click();
      await expect(page.locator('#results-panel')).toBeVisible();

      const layout = await page.evaluate(() => {
        const panel = document.querySelector('#results-panel .section-body').getBoundingClientRect();
        const cards = [...document.querySelectorAll('.model')].map((c) => c.getBoundingClientRect());
        return {
          scrollWidth: document.documentElement.scrollWidth,
          innerWidth: window.innerWidth,
          cardsInside: cards.every((c) => c.right <= panel.right + 0.5),
          stacked: cards[1].top >= cards[0].bottom,
        };
      });
      expect(layout.scrollWidth).toBeLessThanOrEqual(layout.innerWidth);
      expect(layout.cardsInside).toBe(true);
      expect(layout.stacked).toBe(true);
    });
  }
});

test.describe('accessibility', () => {
  // axe is injected as an inline script, which the page's CSP correctly blocks.
  test.use({ bypassCSP: true });

  async function expectNoViolations(page) {
    await page.addScriptTag({ path: require.resolve('axe-core') });
    // eslint-disable-next-line no-undef
    const violations = await page.evaluate(async () => (await axe.run()).violations);
    expect(violations.map((v) => `${v.id}: ${v.help} (${v.nodes.length})`)).toEqual([]);
  }

  test('initial page', async ({ page }) => {
    await page.goto('/');
    await expectNoViolations(page);
  });

  test('with validation errors', async ({ page }) => {
    await page.goto('/');
    await page.getByRole('button', { name: 'Analyze article' }).click();
    await expectNoViolations(page);
  });

  test('with results', async ({ page }) => {
    await mockPredict(page, 200, { ml: prediction('REAL', 0.69), ai: prediction('UNAVAILABLE', null) });
    await page.goto('/');
    await fillForm(page);
    await page.getByRole('button', { name: 'Analyze article' }).click();
    await expect(page.locator('#results-panel')).toBeVisible();
    await expectNoViolations(page);
  });
});

test('static assets are served', async ({ request }) => {
  for (const path of ['/style.css', '/app.js', '/favicon.svg']) {
    expect((await request.get(path)).status(), path).toBe(200);
  }
});
