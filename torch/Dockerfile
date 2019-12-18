# FROM pytorch/pytorch:latest
FROM pytorch/pytorch:1.3-cuda10.1-cudnn7-runtime

WORKDIR /workspace/torch
COPY torch/requirements.txt /workspace/torch
RUN pip install -r requirements.txt