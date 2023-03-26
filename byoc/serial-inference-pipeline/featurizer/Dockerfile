FROM python:3.9-slim-buster

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    ca-certificates \
    wget \
    nginx \
    && rm -rf /var/lib/apt/lists/*

ARG PYTHON=python3
ARG PYTHON_PIP=python3-pip
ARG PIP=pip3

RUN ${PIP} --no-cache-dir install --upgrade pip

COPY requirements.txt /opt/program/requirements.txt
RUN pip install --no-cache-dir -r /opt/program/requirements.txt

# COPY inference.py /opt/program/inference.py
COPY code/ /opt/program/

WORKDIR /opt/program

EXPOSE 8080

ENTRYPOINT ["python"]

CMD [ "serve" ]

