FROM python:3.7

RUN pip install -U pip setuptools wheel

WORKDIR /app
COPY requirements.txt /app
RUN pip install -r requirements.txt
RUN python -c "import tensorflow_hub as hub; hub.Module('https://tfhub.dev/google/elmo/3', trainable=True)"
RUN spacy download en

COPY neuraleduseg neuraleduseg
COPY app app
COPY build /build
COPY setup.py .
COPY scripts scripts
RUN pip install -e .

EXPOSE 8080
ENTRYPOINT python app/api.py 0.0.0.0 --port 8080 --model_dir /build/models