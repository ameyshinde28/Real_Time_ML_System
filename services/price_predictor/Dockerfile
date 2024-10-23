# Let's start with a python 3.10 slim image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install build-essential which includes gcc and other build tools, and git
RUN apt-get update && apt-get install -y build-essential wget git

RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib/ && \
    ./configure --prefix=/usr --build=aarch64-unknown-linux-gnu && \
    make && \
    make install && \
    cd .. && \
    rm -rf ta-lib*

# Install Python Poetry with version 1.8.3
RUN pip install poetry==1.8.3

# Copy only the pyproject.toml and poetry.lock files to leverage Docker cache
# COPY pyproject.toml poetry.lock README.md /app/
# Copy the rest of the application code
COPY . /app

# Install dependencies
RUN poetry install

# Expose port 80
EXPOSE 80

# Use an environment variable to determine which script to run
CMD if [ "$TRAINING_OR_PREDICTION" = "training" ]; then \
        poetry run python src/training.py; \
    elif [ "$TRAINING_OR_PREDICTION" = "prediction" ]; then \
        poetry run uvicorn src.api:app --host 0.0.0.0 --port 80; \
    else \
        echo "Invalid TRAINING_OR_PREDICTION. Please set TRAINING_OR_PREDICTION to 'training' or 'prediction'." && exit 1; \
    fi