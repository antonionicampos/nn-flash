FROM tensorflow/tensorflow:2.16.1-gpu

LABEL maintainer "Antonioni Barros Campos <antonioni.campos@petrobras.com.br>"
LABEL version "0.0.1"

WORKDIR /nn-flash

COPY ./ ./

RUN python3 -m pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt