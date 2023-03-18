# Part of the implementation of this container is based on the Amazon SageMaker Apache MXNet container.
# https://github.com/aws/sagemaker-mxnet-container

# FROM ubuntu:18.04
FROM python:3.9-slim-buster

LABEL maintainer="Amazon AI"

# Defining some variables used at build time to install Python3
ARG PYTHON=python3
ARG PYTHON_PIP=python3-pip
ARG PIP=pip3
# ARG PYTHON_VERSION=3.9.16

# Install some handful libraries like curl, wget, git, build-essential, zlib
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    python3-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Installing Python3
# RUN wget https://www.python.org/ftp/python/$PYTHON_VERSION/Python-$PYTHON_VERSION.tgz && \
#         tar -xvf Python-$PYTHON_VERSION.tgz && cd Python-$PYTHON_VERSION && \
#         ./configure && make && make install && \
#         apt-get update && apt-get install -y --no-install-recommends libreadline-gplv2-dev libncursesw5-dev libssl-dev libsqlite3-dev tk-dev libgdbm-dev libc6-dev libbz2-dev && \
#         make && make install && rm -rf ../Python-$PYTHON_VERSION* && \
#         ln -s /usr/local/bin/pip3 /usr/bin/pip

# Upgrading pip and creating symbolic link for python3
RUN ${PIP} --no-cache-dir install --upgrade pip
# RUN ln -s $(which ${PYTHON}) /usr/local/bin/python

WORKDIR /

# Installing numpy, pandas, scikit-learn, scipy
RUN ${PIP} install --no-cache --upgrade \
        numpy \
        pandas \
        scikit-learn \
        xgboost \
        sagemaker-training

# Setting some environment variables.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/lib" \
    PYTHONIOENCODING=UTF-8 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

COPY code/* /

ENTRYPOINT ["python", "main.py"]