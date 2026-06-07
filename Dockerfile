FROM python:3.11-slim

# Create non-root user
RUN groupadd --gid 1001 app && \
    useradd --uid 1001 --gid app --no-create-home app

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY model ./model

RUN chown -R app:app /app

USER app

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')"

CMD ["uvicorn","app.main:app","--host","0.0.0.0","--port","8080"]