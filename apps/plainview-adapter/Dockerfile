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
RUN groupadd --system --gid 65534 heli || true \
 && useradd  --system --uid 65534 --gid 65534 --no-create-home heli || true \
 && chown -R heli:heli /app
USER heli:heli
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8005"]
