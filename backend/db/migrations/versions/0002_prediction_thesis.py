"""add thesis column to predictions

Revision ID: 0002_prediction_thesis
Revises: 0001_initial
Create Date: 2026-05-22

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "0002_prediction_thesis"
down_revision: Union[str, None] = "0001_initial"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("predictions", sa.Column("thesis", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("predictions", "thesis")
