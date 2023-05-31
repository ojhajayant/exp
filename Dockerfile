FROM ubuntu:latest

WORKDIR /workspace
COPY train.py /workspace/

CMD ["python", "train.py"]