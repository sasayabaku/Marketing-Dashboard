FROM python:3.9.12-slim-buster

COPY requirements.txt /requirements.txt

RUN apt update && apt install -y gcc

RUN pip install --upgrade pip
RUN pip install -r /requirements.txt

# RUN pip install scipy==1.8.0 scikit-learn==1.0.2
