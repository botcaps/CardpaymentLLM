FROM python:3.11-slim

WORKDIR /app

# System deps: gcc/g++ for compiled packages, git for pip installs from git
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first (layer cached until requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Streamlit server config (headless, no browser, correct port)
RUN mkdir -p /root/.streamlit
RUN printf '[server]\nheadless = true\naddress = "0.0.0.0"\nport = 8501\nenableCORS = false\nenableXsrfProtection = false\n' \
    > /root/.streamlit/config.toml

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "dashboard.py", \
    "--server.port=8501", \
    "--server.address=0.0.0.0", \
    "--server.headless=true"]
