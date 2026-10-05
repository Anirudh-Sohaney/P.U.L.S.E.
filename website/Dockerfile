FROM python:3.14-slim-trixie

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

COPY requirements-prod.lock .
RUN pip install --no-cache-dir --only-binary=:all: -r requirements-prod.lock

# Copy application code
COPY . .
RUN groupadd --gid 10001 pulse && useradd --uid 10001 --gid pulse --no-create-home --shell /usr/sbin/nologin pulse \
    && mkdir -p /app/data && chown pulse:pulse /app/data
USER pulse

# Expose port
EXPOSE 8000

# Run the application
CMD ["python", "-m", "backend.main"]
