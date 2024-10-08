# Let's start with a python 3.12.3 slim image
FROM python:3.12.3-slim

# Set working directory in the container
WORKDIR /app

# Install build-essential which includes gcc and other build tools
RUN apt-get update && apt-get install -y build-essential

# Install Python poetry
RUN pip install poetry==1.8.3

# Copy the source code to the working directory
COPY . /app

#Install dependencies
RUN poetry install

# Run the appilicaion
CMD [ "poetry", "run", "python", "src/main.py"]