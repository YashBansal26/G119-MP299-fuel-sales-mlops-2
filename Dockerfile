FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# install libraries first, so this layer is cached when only code changes
COPY requirements.txt .
RUN pip install -r requirements.txt

# then the code and the champion model
COPY app/ app/
COPY src/ src/
COPY models/model.joblib models/model.joblib
COPY params.yaml .

ENV PYTHONPATH=/app/src:/app

RUN useradd --create-home appuser
USER appuser

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
