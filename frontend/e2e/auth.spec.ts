import { test, expect } from '@playwright/test';

test.describe('Authentication Flow', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
  });

  test('should display login form by default', async ({ page }) => {
    await expect(page.getByPlaceholder(/username/i)).toBeVisible();
    await expect(page.getByPlaceholder(/password/i)).toBeVisible();
    await expect(page.getByRole('button', { name: /sign in/i })).toBeVisible();
  });

  test('should toggle between login and register modes', async ({ page }) => {
    await page.getByRole('button', { name: /don't have an account/i }).click();
    await expect(page.getByText(/join nova/i)).toBeVisible();
    await expect(page.getByPlaceholder(/create password/i)).toBeVisible();
    
    await page.getByRole('button', { name: /already have an account/i }).click();
    await expect(page.getByText(/welcome back/i)).toBeVisible();
  });

  test('should show validation error for empty fields', async ({ page }) => {
    const form = page.getByPlaceholder(/username/i).locator('..').locator('form');
    await form.evaluate((form: HTMLFormElement) => form.submit());
    
    await expect(page.getByText(/username and password are required/i)).toBeVisible();
  });

  test('should handle login with valid credentials', async ({ page }) => {
    // Mock the API response
    await page.route('**/api/v1/auth/token', async route => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          access_token: 'mock-token-123',
          token_type: 'bearer'
        })
      });
    });

    await page.getByPlaceholder(/username/i).fill('testuser');
    await page.getByPlaceholder(/password/i).fill('testpass123');
    await page.getByRole('button', { name: /sign in/i }).click();

    // Should redirect to dashboard after successful login
    await expect(page).toHaveURL(/\/dashboard/);
  });

  test('should show error message for invalid credentials', async ({ page }) => {
    await page.route('**/api/v1/auth/token', async route => {
      await route.fulfill({
        status: 401,
        contentType: 'application/json',
        body: JSON.stringify({ detail: 'Invalid credentials' })
      });
    });

    await page.getByPlaceholder(/username/i).fill('wronguser');
    await page.getByPlaceholder(/password/i).fill('wrongpass');
    await page.getByRole('button', { name: /sign in/i }).click();

    await expect(page.getByText(/invalid credentials/i)).toBeVisible();
  });

  test('should handle user registration', async ({ page }) => {
    await page.route('**/api/v1/auth/register', async route => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          username: 'newuser',
          message: 'Registration successful'
        })
      });
    });

    await page.getByRole('button', { name: /don't have an account/i }).click();
    await page.getByPlaceholder(/username/i).fill('newuser');
    await page.getByPlaceholder(/create password/i).fill('securepass123');
    await page.getByRole('button', { name: /create account/i }).click();

    await expect(page.getByText(/registration successful/i)).toBeVisible();
  });

  test('should disable form during submission', async ({ page }) => {
    // Mock a slow response
    await page.route('**/api/v1/auth/token', async route => {
      await new Promise(resolve => setTimeout(resolve, 2000));
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          access_token: 'mock-token',
          token_type: 'bearer'
        })
      });
    });

    await page.getByPlaceholder(/username/i).fill('testuser');
    await page.getByPlaceholder(/password/i).fill('testpass');
    await page.getByRole('button', { name: /sign in/i }).click();

    // Button should be disabled during loading
    await expect(page.getByRole('button', { name: /sign in/i })).toBeDisabled();
  });
});
