# Database Migrations

Versioned SQL migrations using [Flyway](https://flywaydb.org/) naming convention (`V{version}__{description}.sql`). Safe to run on a fresh database — all statements are idempotent (`CREATE TABLE IF NOT EXISTS`, `CREATE INDEX IF NOT EXISTS`).

## Files

| File | Description |
|------|-------------|
| `V1__initial_schema.sql` | Core star schema: tenants, API keys, users, movies (SCD2), fact events |
| `V2__nova_content_events.sql` | Nova content events table used by the JSONL/Postgres event store |
| `V3__experiment_results.sql` | A/B experiment results snapshot table |

## Running with Flyway (Docker)

```bash
docker run --rm \
  -e FLYWAY_URL=jdbc:postgresql://localhost:5432/nova_db \
  -e FLYWAY_USER=nova_user \
  -e FLYWAY_PASSWORD=nova_password \
  -v $(pwd)/sql/migrations:/flyway/sql \
  flyway/flyway:10 migrate
```

## Running manually (psql)

```bash
psql $DATABASE_URL -f sql/migrations/V1__initial_schema.sql
psql $DATABASE_URL -f sql/migrations/V2__nova_content_events.sql
psql $DATABASE_URL -f sql/migrations/V3__experiment_results.sql
```

## Adding a new migration

1. Create `V{N+1}__{short_description}.sql` in this directory
2. Use `IF NOT EXISTS` for all DDL statements
3. Never modify existing migration files — add a new one instead
