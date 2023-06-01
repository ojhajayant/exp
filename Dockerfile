FROM python:3.9-slim

WORKDIR /workspace

RUN pip3 install --no-cache-dir torch==1.9.0+cpu torchvision==0.10.0+cpu --index-url https://download.pytorch.org/whl/cpu \
    && pip3 install --no-cache-dir torchsummary==1.5.1 \
    && pip3 install --no-cache-dir tqdm==4.62.3 \
    && rm -rf /root/.cache/pip


COPY train.py /workspace

CMD ["python", "train.py"]
