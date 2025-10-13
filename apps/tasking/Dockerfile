FROM python:3.11-slim

WORKDIR /app

ARG INCLUDE_PYMAVLINK=false

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
    && if [ "$INCLUDE_PYMAVLINK" = "true" ]; then pip install --no-cache-dir pymavlink; fi

COPY . .

EXPOSE 8004

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8004", "--reload"]
