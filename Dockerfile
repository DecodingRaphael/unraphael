# Build:
# docker build -t ghcr.io/decodingraphael/unraphael .
# Push to GitHub Container Registry:
# docker push ghcr.io/decodingraphael/unraphael
# Run:
# docker run -p 8501:8501 ghcr.io/decodingraphael/unraphael
FROM python:3.12-slim

RUN pip install torch==2.3.1+cpu torchvision torchaudio 'numpy<2.0' --extra-index-url https://download.pytorch.org/whl/cpu
RUN apt-get update && apt-get install libgl1 libglib2.0-0 -y

WORKDIR /app

COPY pyproject.toml MANIFEST.in ./
COPY ./src ./src

RUN pip install .[dash]

EXPOSE 8501

ENTRYPOINT ["unraphael-dash", "--server.port=8501"]
CMD ["--server.address=0.0.0.0"]
