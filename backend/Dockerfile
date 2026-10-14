# Build from the REPO ROOT so the image can carry ml/results/cv_metrics.json,
# which the /metrics endpoint reads from disk:
#   docker build -f backend/Dockerfile -t cardiovision-api .
FROM python:3.12-slim

WORKDIR /srv

COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/app ./backend/app
COPY ml/results ./ml/results

ENV PYTHONUNBUFFERED=1
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health').status==200 else 1)"

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
