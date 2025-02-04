FROM ubuntu:22.04

# Set environment variables to avoid user interaction during package installation
ENV DEBIAN_FRONTEND=noninteractive

ENV LIBGL_ALWAYS_INDIRECT=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    python3-setuptools \
    python3-pyqt5 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy the project files into the Docker image
COPY . /app

# Install Python dependencies
RUN pip install -r requirements.txt

ENV DISPLAY=:0
ENV QT_DEBUG_PLUGINS=0

# Set the entrypoint to run the application
ENTRYPOINT ["python3", "main/main.py"]