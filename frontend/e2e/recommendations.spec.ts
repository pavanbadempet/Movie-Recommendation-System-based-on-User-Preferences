import { test, expect } from '@playwright/test';

test.describe('Recommendations Flow', () => {
  test.beforeEach(async ({ page }) => {
    // Mock authentication
    await page.goto('/');
    await page.evaluate(() => {
      localStorage.setItem('token', 'mock-token');
      localStorage.setItem('user', JSON.stringify({ username: 'testuser' }));
    });
  });

  test('should display movie recommendations on dashboard', async ({ page }) => {
    await page.goto('/dashboard');
    
    // Mock the recommendations API
    await page.route('**/api/v1/recommendations/user/**', async route => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify([
          {
            id: 19995,
            title: 'Avatar',
            poster_path: '/jRXYjXNq0Cs2TcJjLkki24MLp7u.jpg',
            vote_average: 7.6,
            genres: 'Action, Adventure, Fantasy',
            release_date: '2009-12-18',
            overview: 'A paraplegic Marine dispatched to the moon Pandora.'
          },
          {
            id: 155,
            title: 'The Dark Knight',
            poster_path: '/qJ2tW6WMUDux911r6m7haRef0WH.jpg',
            vote_average: 8.5,
            genres: 'Action, Crime, Drama',
            release_date: '2008-07-18',
            overview: 'Batman raises the stakes in his war on crime.'
          }
        ])
      });
    });

    await expect(page.getByText('Avatar')).toBeVisible();
    await expect(page.getByText('The Dark Knight')).toBeVisible();
  });

  test('should navigate to movie details when clicking a movie', async ({ page }) => {
    await page.goto('/dashboard');
    
    await page.route('**/api/v1/recommendations/user/**', async route => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify([
          {
            id: 19995,
            title: 'Avatar',
            poster_path: '/jRXYjXNq0Cs2TcJjLkki24MLp7u.jpg',
            vote_average: 7.6,
            genres: 'Action, Adventure, Fantasy',
            release_date: '2009-12-18',
            overview: 'A paraplegic Marine dispatched to the moon Pandora.'
          }
        ])
      });
    });

    await page.getByText('Avatar').click();
    
    // Should navigate to movie details or show modal
    await expect(page.getByText('A paraplegic Marine dispatched to the moon Pandora')).toBeVisible();
  });

  test('should handle search functionality', async ({ page }) => {
    await page.goto('/dashboard');
    
    await page.route('**/api/v1/search**', async route => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify([
          {
            id: 550,
            title: 'Fight Club',
            poster_path: '/pB8BM7pdSp6B6Ih7QZ4DrQ3PmJK.jpg',
            vote_average: 8.4,
            genres: 'Drama',
            release_date: '1999-10-15'
          }
        ])
      });
    });

    const searchInput = page.getByPlaceholder(/search/i);
    await searchInput.fill('Fight Club');
    await searchInput.press('Enter');

    await expect(page.getByText('Fight Club')).toBeVisible();
  });

  test('should handle visually similar recommendations', async ({ page }) => {
    await page.goto('/dashboard');
    
    await page.route('**/api/v1/recommendations/visually-similar/**', async route => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify([
          {
            id: 19995,
            title: 'Avatar',
            poster_path: '/jRXYjXNq0Cs2TcJjLkki24MLp7u.jpg',
            vote_average: 7.6,
            genres: 'Action, Adventure, Fantasy'
          }
        ])
      });
    });

    // Navigate to a movie and click visually similar
    await page.getByText('Avatar').click();
    await page.getByRole('button', { name: /visually similar/i }).click();

    await expect(page.getByText('Avatar')).toBeVisible();
  });

  test('should handle knowledge graph recommendations', async ({ page }) => {
    await page.goto('/dashboard');
    
    await page.route('**/api/v1/recommendations/knowledge-graph/**', async route => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify([
          {
            id: 155,
            title: 'The Dark Knight',
            poster_path: '/qJ2tW6WMUDux911r6m7haRef0WH.jpg',
            vote_average: 8.5,
            genres: 'Action, Crime, Drama'
          }
        ])
      });
    });

    await page.getByText('Avatar').click();
    await page.getByRole('button', { name: /knowledge graph/i }).click();

    await expect(page.getByText('The Dark Knight')).toBeVisible();
  });

  test('should handle loading state for recommendations', async ({ page }) => {
    await page.goto('/dashboard');
    
    // Mock a slow response
    await page.route('**/api/v1/recommendations/user/**', async route => {
      await new Promise(resolve => setTimeout(resolve, 2000));
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify([])
      });
    });

    await expect(page.getByText(/loading/i)).toBeVisible();
  });

  test('should handle error state for recommendations', async ({ page }) => {
    await page.goto('/dashboard');
    
    await page.route('**/api/v1/recommendations/user/**', async route => {
      await route.fulfill({
        status: 500,
        contentType: 'application/json',
        body: JSON.stringify({ detail: 'Internal server error' })
      });
    });

    await expect(page.getByText(/error/i)).toBeVisible();
  });
});
