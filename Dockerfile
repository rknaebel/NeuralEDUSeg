FROM python:3.7

RUN pip install -U pip setuptools wheel

WORKDIR /app
COPY requirements.txt /app
RUN pip install -r requirements.txt
RUN python -c "import tensorflow_hub as hub; hub.Module('https://tfhub.dev/google/elmo/3', trainable=True)"
RUN spacy download en

COPY neuraleduseg neuraleduseg
COPY setup.py .
COPY scripts scripts
RUN pip install -e .

ENTRYPOINT bash scripts/run.sh