FROM tensorflow/tensorflow:2.16.1-gpu

LABEL maintainer="Antonioni Barros Campos <antonioni.campos@petrobras.com.br>"
LABEL version="0.0.1"

# Install latex and dependencies
# Needed to generate tables and plots
# Reference: https://www.joshfinnie.com/blog/latex-through-docker/
RUN apt-get update && \
    apt-get install --no-install-recommends -y \ 
        biber \ 
        latexmk \ 
        texlive-full && \
        rm -rf /var/lib/apt/lists/*

# Install OpenJDK 11
# Needed to run neqsim package
# Reference: https://stackoverflow.com/a/61713897
RUN apt-get update && \ 
    apt-get install -y openjdk-11-jre-headless && \ 
    apt-get clean;

WORKDIR /nn-flash

COPY ./src ./src
COPY ./main.py ./main.py
COPY ./requirements.txt ./requirements.txt

RUN python3 -m pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt