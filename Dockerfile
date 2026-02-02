FROM python:3.11-slim

WORKDIR /app

# System deps (minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code only (data files mounted at runtime)
COPY config.py precompute.py app.py ./
COPY data/ data/
COPY analytics/ analytics/
COPY viz/ viz/
COPY _pages/ _pages/
COPY logo.png ./

# Streamlit config
RUN mkdir -p /root/.streamlit
RUN printf '[server]\nheadless = true\nport = 8501\nenableCORS = false\nenableXsrfProtection = false\nmaxUploadSize = 500\n\n[browser]\ngatherUsageStats = false\n' > /root/.streamlit/config.toml

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Data files are expected at /app/all_trackman.parquet and /app/davidson.duckdb
# Mount them:  docker run -v /path/to/data:/app/data_mount ...
# Or place them in the build context before building.

ENTRYPOINT ["streamlit", "run", "app.py"]
