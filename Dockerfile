FROM python:3.12 as builder

RUN pip install torch==2.3.1+cpu torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
# RUN apt-get update && apt-get install libsm6 libxext6  -y
RUN apt-get update && apt-get install libgl1 -y

WORKDIR /app

COPY pyproject.toml MANIFEST.in ./
COPY ./src ./src

RUN pip install .[dash]

EXPOSE 8501

ENTRYPOINT ["unraphael-dash", "--server.port=8501", "--server.address=0.0.0.0"]
