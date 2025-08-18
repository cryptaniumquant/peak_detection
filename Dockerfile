FROM python:3.11-slim-bookworm

# Install system dependencies
RUN apt-get update \
 && apt-get -y install build-essential openssh-client \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# Set timezone to Moscow
RUN echo "Europe/Moscow" > /etc/timezone \
    && ln -fs /usr/share/zoneinfo/Europe/Moscow /etc/localtime \
    && dpkg-reconfigure -f noninteractive tzdata

# Set working directory
WORKDIR /root/app

# Copy requirements first (rarely changes - better caching)
COPY bot_service/requirements.txt ./bot_service/

# Upgrade pip and install Python dependencies
RUN python -m pip install --upgrade pip \
    && pip install -r ./bot_service/requirements.txt

# Set environment variable for Docker detection
ENV DOCKER_ENV=1

# Default command - run bot as module
CMD ["python", "-m", "bot_service.run_bot"]

# Copy the entire project (changes frequently - last layer)
COPY . ./
