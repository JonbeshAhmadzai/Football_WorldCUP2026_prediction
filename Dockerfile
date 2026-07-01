FROM node:20-slim AS frontend-builder

WORKDIR /app

COPY app/frontend/package*.json app/frontend/
WORKDIR /app/app/frontend
RUN npm ci

WORKDIR /app
COPY app/frontend app/frontend
RUN cd app/frontend && npm run build


FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV PORT=8501

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY . .
COPY --from=frontend-builder /app/app/frontend/dist /app/app/frontend/dist

EXPOSE 8501

CMD ["sh", "-c", "uvicorn app.backend.main:app --host 0.0.0.0 --port ${PORT:-8501}"]
