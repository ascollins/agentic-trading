FROM python:3.11-slim AS base

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml .
COPY src/ src/
COPY configs/ configs/
COPY scripts/ scripts/
COPY alembic.ini .
COPY alembic/ alembic/

# Install Python dependencies
RUN pip install --no-cache-dir -e ".[dev]"

# Create data directories
RUN mkdir -p data/historical data/logs

# Non-root user
RUN useradd -m trader
USER trader

# Default command
ENTRYPOINT ["python", "-m", "agentic_trading.cli"]
CMD ["backtest", "--config", "configs/backtest.toml"]
