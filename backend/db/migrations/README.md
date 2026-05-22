# Alembic Migrations

Database migrations for the StockSage 2.0 schema.

Alembic is already initialised (`env.py`, `script.py.mako`, and the initial
migration are committed). The database URL is injected at runtime from
`Settings.DATABASE_URL` — you do not need to edit `alembic.ini`.

## Apply migrations

```bash
# From the project root, with DATABASE_URL set (or a .env file present):
alembic upgrade head
```

`0001_initial` creates all core tables. If the target database has the
TimescaleDB extension available, `price_data` is automatically converted to a
hypertable; on plain PostgreSQL it remains a regular table.

## Create a new migration

```bash
# Autogenerate from ORM model changes in backend/db/models.py:
alembic revision --autogenerate -m "describe the change"

# Review the generated file in versions/, then apply it:
alembic upgrade head
```

## Local dev shortcut

For local development you can skip Alembic entirely by setting
`DB_AUTO_CREATE=true` in `.env` — tables are then created from the ORM models
on app startup. Production should always use `alembic upgrade head`.
