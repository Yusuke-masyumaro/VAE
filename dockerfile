FROM nvcr.io/nvidia/pytorch:23.10-py3

RUN apt update && apt upgrade -y 
RUN mkdir /app
RUN pip install -U pip
RUN pip install torchaudio
