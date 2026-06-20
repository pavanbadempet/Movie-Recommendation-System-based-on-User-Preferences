# Frontend Architecture Strategy

## Current State Analysis

The project currently has two frontend implementations:
1. **React + Vite** (`frontend/`) - Modern, production-ready SPA
2. **Streamlit** (`frontend/streamlit_app.py`) - Rapid prototyping tool (97KB)

## Recommended Architecture: Unified React Frontend

### Decision: Deprecate Streamlit, Standardize on React

**Rationale**:
- Streamlit is excellent for prototyping but not production-grade
- React provides better UX, performance, and maintainability
- Dual frontend increases maintenance burden
- React ecosystem offers superior component libraries and tooling
- Better separation of concerns (API vs UI)

### Migration Strategy

#### Phase 1: Feature Parity Assessment
Identify Streamlit features not yet in React:
- [ ] Admin dashboard visualization
- [ ] Real-time recommendation testing
- [ ] Model performance charts
- [ ] User behavior analytics

#### Phase 2: React Component Development
Create React equivalents for Streamlit features:

```typescript
// frontend/src/components/AdminDashboard.tsx
import React, { useState, useEffect } from 'react';
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from 'recharts';

interface DashboardProps {
  apiEndpoint: string;
}

export const AdminDashboard: React.FC<DashboardProps> = ({ apiEndpoint }) => {
  const [metrics, setMetrics] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch(`${apiEndpoint}/v1/platform/slo`)
      .then(res => res.json())
      .then(data => {
        setMetrics(data);
        setLoading(false);
      });
  }, [apiEndpoint]);

  if (loading) return <div>Loading dashboard...</div>;

  return (
    <div className="admin-dashboard">
      <h2>System Performance</h2>
      <LineChart width={600} height={300} data={metrics.latency_history}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="timestamp" />
        <YAxis />
        <Tooltip />
        <Legend />
        <Line type="monotone" dataKey="latency_ms" stroke="#8884d8" />
      </LineChart>
      
      <h3>Model Performance</h3>
      <BarChart width={600} height={300} data={metrics.model_metrics}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="model" />
        <YAxis />
        <Tooltip />
        <Legend />
        <Bar dataKey="hr@10" fill="#8884d8" />
        <Bar dataKey="ndcg@10" fill="#82ca9d" />
      </BarChart>
    </div>
  );
};
```

#### Phase 3: Streamlit Deprecation
- Add deprecation notice to Streamlit app
- Document migration path in README
- Remove Streamlit dependencies in next major version

### New Frontend Structure

```
frontend/
├── src/
│   ├── components/
│   │   ├── MovieCard.tsx          # Movie display component
│   │   ├── RecommendationList.tsx # Main recommendation UI
│   │   ├── SearchBar.tsx           # Search functionality
│   │   ├── AdminDashboard.tsx      # Admin analytics (NEW)
│   │   ├── ModelExplorer.tsx       # Model testing (NEW)
│   │   └── UserProfile.tsx         # User preferences
│   ├── pages/
│   │   ├── Home.tsx                # Main landing page
│   │   ├── MovieDetail.tsx         # Individual movie page
│   │   ├── Admin.tsx               # Admin panel (NEW)
│   │   └── Settings.tsx            # User settings
│   ├── hooks/
│   │   ├── useRecommendations.ts   # Recommendation API hook
│   │   ├── useAuth.ts              # Authentication hook
│   │   └── useAnalytics.ts         # Analytics tracking
│   ├── services/
│   │   ├── api.ts                  # API client
│   │   └── websocket.ts            # Real-time updates (NEW)
│   ├── types/
│   │   ├── movie.ts                # Movie types
│   │   ├── recommendation.ts       # Recommendation types
│   │   └── user.ts                 # User types
│   ├── utils/
│   │   ├── formatters.ts           # Data formatting
│   │   └── validators.ts           # Input validation
│   └── App.tsx
├── public/
└── package.json
```

> **Status:** This document is a design proposal. Features in roadmap sections are not current implementation claims.

### Enhanced React Features

#### 1. Real-time Updates with WebSocket
```typescript
// frontend/src/services/websocket.ts
class RecommendationWebSocket {
  private ws: WebSocket | null = null;
  
  connect(userId: string) {
    this.ws = new WebSocket(`ws://localhost:8000/ws/${userId}`);
    
    this.ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      // Update recommendations in real-time
      this.handleRecommendationUpdate(data);
    };
  }
  
  disconnect() {
    this.ws?.close();
  }
}
```

#### 2. Advanced State Management
```typescript
// frontend/src/hooks/useRecommendations.ts
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';

export const useRecommendations = (movieId: number) => {
  return useQuery({
    queryKey: ['recommendations', movieId],
    queryFn: () => fetch(`/api/v1/recommendations/id/${movieId}`).then(r => r.json()),
    staleTime: 5 * 60 * 1000, // 5 minutes
    cacheTime: 10 * 60 * 1000, // 10 minutes
  });
};

export const useRateMovie = () => {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: (rating: { movieId: number; rating: number }) =>
      fetch('/api/v1/events', {
        method: 'POST',
        body: JSON.stringify(rating),
      }).then(r => r.json()),
    onSuccess: () => {
      queryClient.invalidateQueries(['recommendations']);
    },
  });
};
```

#### 3. Performance Optimizations
```typescript
// frontend/src/components/MovieCard.tsx
import React, { memo } from 'react';

export const MovieCard = memo(({ movie }: { movie: Movie }) => {
  // Component only re-renders when movie prop changes
  return (
    <div className="movie-card">
      <img src={movie.poster} alt={movie.title} loading="lazy" />
      <h3>{movie.title}</h3>
      <p>{movie.genres.join(', ')}</p>
    </div>
  );
});
```

### Component Library Integration

Install shadcn/ui for consistent, accessible components:

```bash
cd frontend
npx shadcn-ui@latest init
npx shadcn-ui@latest add button
npx shadcn-ui@latest add card
npx shadcn-ui@latest add input
npx shadcn-ui@latest add dialog
npx shadcn-ui@latest add dropdown-menu
```

### Updated Package.json Dependencies

```json
{
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "react-router-dom": "^6.20.0",
    "@tanstack/react-query": "^5.0.0",
    "recharts": "^2.10.0",
    "lucide-react": "^0.300.0",
    "clsx": "^2.0.0",
    "tailwind-merge": "^2.0.0"
  },
  "devDependencies": {
    "@types/react": "^18.2.0",
    "@types/react-dom": "^18.2.0",
    "@vitejs/plugin-react": "^4.2.0",
    "typescript": "^5.3.0",
    "vite": "^5.0.0",
    "tailwindcss": "^3.4.0",
    "autoprefixer": "^10.4.0",
    "postcss": "^8.4.0"
  }
}
```

### Testing Strategy

```typescript
// frontend/src/components/__tests__/MovieCard.test.tsx
import { render, screen } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import { MovieCard } from '../MovieCard';

describe('MovieCard', () => {
  it('renders movie title', () => {
    const movie = { id: 1, title: 'Test Movie', genres: ['Action'], poster: 'url' };
    render(<MovieCard movie={movie} />);
    expect(screen.getByText('Test Movie')).toBeInTheDocument();
  });
  
  it('displays genres correctly', () => {
    const movie = { id: 1, title: 'Test', genres: ['Action', 'Drama'], poster: 'url' };
    render(<MovieCard movie={movie} />);
    expect(screen.getByText('Action, Drama')).toBeInTheDocument();
  });
});
```

### Performance Targets

| Metric | Target | Current |
|--------|--------|---------|
| Initial Load | <2s | ~3.5s |
| Time to Interactive | <3s | ~4.2s |
| Recommendation Fetch | <500ms | ~800ms |
| Bundle Size | <500KB | ~650KB |

### Migration Timeline

- **Week 1**: Create React admin dashboard components
- **Week 2**: Implement WebSocket real-time updates
- **Week 3**: Add advanced state management with React Query
- **Week 4**: Performance optimization and testing
- **Week 5**: Streamlit deprecation notice and documentation
- **Week 6**: Remove Streamlit dependencies

### Rollback Plan

If issues arise during migration:
1. Keep Streamlit app as `frontend/legacy_streamlit_app.py`
2. Maintain feature parity documentation
3. Provide fallback endpoints for critical features
4. Monitor user feedback and performance metrics

### Success Metrics

- [ ] All Streamlit features available in React
- [ ] React app performance meets targets
- [ ] User testing shows improved satisfaction
- [ ] Reduced maintenance overhead
- [ ] Improved code consistency

## Conclusion

The unified React frontend provides:
- **Better Performance**: Modern React optimizations
- **Improved UX**: Consistent, responsive design
- **Easier Maintenance**: Single codebase, modern tooling
- **Better Testing**: Comprehensive React testing ecosystem
- **Future-Proof**: Active React ecosystem and community

This proposal describes a possible consolidation path. Completion must be verified against the frontend dependencies, source code, and tests.
