# Reflection - Enterprise Multi-Tenant AI Companion Platform
# Dockerfile for production deployment

# ============================================================
# BUILD STAGE
# ============================================================
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --no-cache-dir --upgrade pip

# Install Reflection (familiar>=1.4.0 pulled automatically via dependency)
COPY pyproject.toml README.md ./
COPY reflection/ ./reflection/
COPY reflection_core/ ./reflection_core/
RUN pip install --no-cache-dir .

# ============================================================
# PRODUCTION STAGE
# ============================================================
FROM python:3.11-slim AS production

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r reflection && useradd -r -g reflection reflection

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY reflection/ ./reflection/
COPY reflection_core/ ./reflection_core/

# Set ownership
RUN chown -R reflection:reflection /app

# Switch to non-root user
USER reflection

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Run application
CMD ["uvicorn", "reflection.gateway.app:app", "--host", "0.0.0.0", "--port", "8000"]
