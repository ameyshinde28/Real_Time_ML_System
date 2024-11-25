# Dockerize cryptyo price predictor
# Let's start with a python 3.12.3 slim image
FROM python:3.12.3-slim

# Set working directory in the container
WORKDIR /app

# Install build-essential which includes gcc and other build tools
RUN apt-get update && apt-get install -y build-essential wget

# Download and install TA-Lib
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib/ && \
    ./configure --prefix=/usr --build=aarch64-unknown-linux-gnu && \
    make && \
    make install && \
    cd .. && \
    rm -rf ta-lib*

# Install Python poetry
RUN pip install poetry==1.8.3

# Copy only dependency files
# COPY pyproject.toml poetry.lock README.md /app/

# Cpoy the rest of the source code
COPY . /app

#Install dependencies
RUN poetry install

EXPOSE 8000

# Run the appilicaion
# Use an environment variable to determine which script to run
CMD if [ "$TRAINING_OR_PREDICTION" = "training" ]; then \
        poetry run python src/training.py; \
    elif [ "$TRAINING_OR_PREDICTION" = "prediction" ]; then \
        poetry run uvicorn src.api:app --host 0.0.0.0 --port 8000; \
    else \
        echo "Invalid TRAINING_OR_PREDICTION. Please set TRAINING_OR_PREDICTION to 'training' or 'prediction'." && exit 1; \
    fi