FROM python:3.9-slim-buster

ARG PYTHON=python3
ARG PYTHON_PIP=python3-pip
ARG PIP=pip3

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    ca-certificates \
    wget \
    nginx \
    && rm -rf /var/lib/apt/lists/*

RUN ${PIP} --no-cache-dir install --upgrade pip

# create /tmp directory if not exists
RUN mkdir -p /tmp

# Copy requirements.txt to /tmp
COPY requirements.txt /tmp/requirements.txt

# Install pypi packages defined in requirements
RUN ${PIP} install --no-cache-dir -r /tmp/requirements.txt

ENV PYTHONUNBUFFERED=TRUE \
    PYTHONDONTWRITEBYTECODE=TRUE \
    PATH="/opt/program:${PATH}"

WORKDIR /opt/program

# Copy all Inference configuration scripts to /opt/program
COPY code /opt/program

ENTRYPOINT [ "python" ]

CMD [ "serve" ]