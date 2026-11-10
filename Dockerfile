# Reproducible Linux environment for the tests and for training runs.
#   docker build -t elt-rl .
#   docker run --rm elt-rl                      # credential-free test suite
#   docker run --rm -e TINKER_API_KEY elt-rl python -m elt_rl.train max_steps=1
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
# CPU torch keeps the image small; Tinker does the GPU work remotely.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY pyproject.toml README.md ./
COPY elt_rl ./elt_rl
RUN pip install --no-cache-dir -e ".[dev,snowflake]"
COPY . .
CMD ["pytest", "-q"]
