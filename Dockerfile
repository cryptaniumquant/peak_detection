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

# Create volumes for SSH keys and keyring (if needed for future extensions)
VOLUME /root/.ssh /root/.local/share/python_keyring

# Set working directory
WORKDIR /root/app

# Copy the entire project
COPY . ./

# Create entrypoint script
COPY entrypoint.sh /root/app/entrypoint.sh
RUN chmod +x /root/app/entrypoint.sh

# Set entrypoint and default command
ENTRYPOINT ["/root/app/entrypoint.sh"]
CMD ["python", "run_bot.py"]
