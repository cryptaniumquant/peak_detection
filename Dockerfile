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

# Upgrade pip and install Python dependencies
COPY bot_service/requirements.txt /root/
RUN python -m pip install --upgrade pip \
    && pip install -r /root/requirements.txt

# Set working directory
WORKDIR /root/app

# Set entrypoint and default command
ENTRYPOINT ["python"]
CMD ["-m", "bot_service.run_bot"]

# Copy the entire project
COPY . ./