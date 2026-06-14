import { test, expect } from '@playwright/test';

test.describe('Admin Panel Flow', () => {
  test.beforeEach(async ({ page }) => {
    // Mock admin authentication
    await page.goto('/');
    await page.evaluate(() => {
      localStorage.setItem('token', 'admin-token');
      localStorage.setItem('user', JSON.stringify({ username: 'admin', role: 'admin' }));
    });
  });

  test('should access admin panel with admin credentials', async ({ page }) => {
    await page.goto('/admin');
    
    await expect(page.getByText(/admin panel/i)).toBeVisible();
    await expect(page.getByText(/ensemble weights/i)).toBeVisible();
  });

  test('should display current ensemble weights', async ({ page }) => {
    await page.goto('/admin');
    
    await page.route('**/api/v1/admin/ensemble-weights', async route => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          sasrec: 0.659,
          kan: 0.298,
          lightgcn: 0.005,
          quantum: 0.010,
          hyperbolic: 0.004,
          diffusion: 0.024
        })
      });
    });

    await expect(page.getByText('SASRec')).toBeVisible();
    await expect(page.getByText('0.659')).toBeVisible();
  });

  test('should reload ensemble weights', async ({ page }) => {
    await page.goto('/admin');
    
    await page.route('**/api/v1/admin/reload-ensemble-weights', async route => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          message: 'Ensemble weights reloaded successfully',
          weights: {
            sasrec: 0.659,
            kan: 0.298,
            lightgcn: 0.005,
            quantum: 0.010,
            hyperbolic: 0.004,
            diffusion: 0.024
          }
        })
      });
    });

    await page.getByRole('button', { name: /reload weights/i }).click();
    
    await expect(page.getByText(/weights reloaded successfully/i)).toBeVisible();
  });

  test('should display system health metrics', async ({ page }) => {
    await page.goto('/admin');
    
    await page.route('**/api/v1/platform/status', async route => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          status: 'healthy',
          uptime: 86400,
          memory_usage: 45.2,
          cpu_usage: 23.1,
          active_connections: 42
        })
      });
    });

    await expect(page.getByText(/healthy/i)).toBeVisible();
    await expect(page.getByText(/uptime/i)).toBeVisible();
  });

  test('should display model evaluation metrics', async ({ page }) => {
    await page.goto('/admin');
    
    await page.route('**/api/v1/evaluation/offline-metrics', async route => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          hr_at_10: 0.785,
          ndcg_at_10: 0.542,
          precision: 0.623,
          recall: 0.451
        })
      });
    });

    await expect(page.getByText('HR@10')).toBeVisible();
    await expect(page.getByText('0.785')).toBeVisible();
  });

  test('should handle weight updates', async ({ page }) => {
    await page.goto('/admin');
    
    await page.route('**/api/v1/admin/update-weights', async route => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          message: 'Weights updated successfully',
          weights: {
            sasrec: 0.700,
            kan: 0.250,
            lightgcn: 0.010,
            quantum: 0.015,
            hyperbolic: 0.005,
            diffusion: 0.020
          }
        })
      });
    });

    // Find and update a weight input
    const sasrecInput = page.getByLabel('SASRec');
    await sasrecInput.clear();
    await sasrecInput.fill('0.700');
    
    await page.getByRole('button', { name: /update weights/i }).click();
    
    await expect(page.getByText(/weights updated successfully/i)).toBeVisible();
  });

  test('should validate weight inputs sum to 1.0', async ({ page }) => {
    await page.goto('/admin');
    
    // Try to set weights that don't sum to 1.0
    const sasrecInput = page.getByLabel('SASRec');
    await sasrecInput.clear();
    await sasrecInput.fill('0.900');
    
    await page.getByRole('button', { name: /update weights/i }).click();
    
    await expect(page.getByText(/weights must sum to 1\.0/i)).toBeVisible();
  });

  test('should handle non-admin access denial', async ({ page }) => {
    // Mock non-admin user
    await page.goto('/');
    await page.evaluate(() => {
      localStorage.setItem('token', 'user-token');
      localStorage.setItem('user', JSON.stringify({ username: 'regularuser', role: 'user' }));
    });
    
    await page.goto('/admin');
    
    await expect(page.getByText(/access denied/i)).toBeVisible();
    await expect(page).toHaveURL('/');
  });
});
