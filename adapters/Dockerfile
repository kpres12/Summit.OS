FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .


# ── Non-root runtime user (for K8s runAsNonRoot enforcement) ──
# UID/GID 65534 = nobody:nogroup on slim images — use numeric IDs directly
RUN chown -R 65534:65534 /app
USER 65534:65534
CMD ["python", "runner.py"]
