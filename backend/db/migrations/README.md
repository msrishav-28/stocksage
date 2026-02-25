# Alembic Migrations

This directory will contain Alembic migration scripts for the TimescaleDB schema.

## Setup

```bash
# Initialize Alembic (one-time)
cd backend
alembic init db/migrations

# Generate migration
alembic revision --autogenerate -m "initial schema"

# Apply migration
alembic upgrade head
```
