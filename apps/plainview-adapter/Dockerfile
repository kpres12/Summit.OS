FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose port
EXPOSE 8005

# Run application

# ── Non-root runtime user (for K8s runAsNonRoot enforcement) ──
# UID/GID 65534 = nobody:nogroup on slim images — use numeric IDs directly
RUN chown -R 65534:65534 /app
USER 65534:65534
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8005"]
