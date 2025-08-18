#!/bin/bash

# Trading Signal Bot - Docker Run Script
# This script builds and runs the trading bot container

set -e

# Configuration
IMAGE_NAME="trading-signal-bot"
CONTAINER_NAME="trading-bot"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=========================================${NC}"
echo -e "${BLUE}🚀 Trading Signal Bot - Docker Runner${NC}"
echo -e "${BLUE}=========================================${NC}"

# Check if local_settings.py exists
if [ ! -f "./local_settings.py" ]; then
    echo -e "${RED}❌ ERROR: local_settings.py not found in current directory!${NC}"
    echo -e "${YELLOW}Please create local_settings.py with your configuration:${NC}"
    echo "  - Database credentials (DB_HOST, DB_USER, etc.)"
    echo "  - Telegram bot token and chat ID"
    echo "  - Other settings as needed"
    exit 1
fi

# Function to stop and remove existing container
cleanup_container() {
    if [ "$(docker ps -aq -f name=${CONTAINER_NAME})" ]; then
        echo -e "${YELLOW}🧹 Stopping and removing existing container...${NC}"
        docker stop ${CONTAINER_NAME} 2>/dev/null || true
        docker rm ${CONTAINER_NAME} 2>/dev/null || true
    fi
}

# Function to build the image
build_image() {
    echo -e "${BLUE}🔨 Building Docker image...${NC}"
    docker build -t ${IMAGE_NAME} .
    echo -e "${GREEN}✅ Image built successfully${NC}"
}

# Function to run the container
run_container() {
    echo -e "${BLUE}🚀 Starting container...${NC}"
    
    # Create host directories for persistent data
    mkdir -p $(pwd)/visualizations
    mkdir -p $(pwd)/bot_service/state
    
    # Create volumes if they don't exist
    docker volume create ssh 2>/dev/null || true
    docker volume create keyring 2>/dev/null || true
    
    # Run the container with host-mounted directories
    docker run -dit \
        --name ${CONTAINER_NAME} \
        --restart unless-stopped \
        --log-opt max-size=100m \
        --log-opt max-file=3 \
        -v $(pwd)/local_settings.py:/root/app/local_settings.py:ro \
        -v $(pwd)/visualizations:/root/app/visualizations \
        -v $(pwd)/bot_service/state:/root/app/bot_service/state \
        --mount source=ssh,destination=/root/.ssh \
        --mount source=keyring,destination=/root/.local/share/python_keyring \
        ${IMAGE_NAME}
    
    echo -e "${GREEN}✅ Container started successfully${NC}"
}

# Function to show logs
show_logs() {
    echo -e "${BLUE}📋 Container logs (last 50 lines):${NC}"
    docker logs --tail 50 ${CONTAINER_NAME}
}

# Function to show status
show_status() {
    echo -e "${BLUE}📊 Container status:${NC}"
    docker ps -f name=${CONTAINER_NAME}
}

# Main execution
case "${1:-run}" in
    "build")
        build_image
        ;;
    "run")
        cleanup_container
        build_image
        run_container
        echo ""
        show_status
        echo ""
        echo -e "${GREEN}🎉 Bot is now running!${NC}"
        echo -e "${YELLOW}Useful commands:${NC}"
        echo "  ./run.sh logs    - Show container logs"
        echo "  ./run.sh status  - Show container status"
        echo "  ./run.sh stop    - Stop the container"
        echo "  ./run.sh restart - Restart the container"
        ;;
    "stop")
        echo -e "${YELLOW}🛑 Stopping container...${NC}"
        docker stop ${CONTAINER_NAME}
        echo -e "${GREEN}✅ Container stopped${NC}"
        ;;
    "restart")
        echo -e "${YELLOW}🔄 Restarting container...${NC}"
        docker restart ${CONTAINER_NAME}
        echo -e "${GREEN}✅ Container restarted${NC}"
        ;;
    "logs")
        show_logs
        ;;
    "status")
        show_status
        ;;
    "shell")
        echo -e "${BLUE}🐚 Opening shell in container...${NC}"
        docker exec -it ${CONTAINER_NAME} /bin/bash
        ;;
    "clean")
        echo -e "${YELLOW}🧹 Cleaning up...${NC}"
        cleanup_container
        docker rmi ${IMAGE_NAME} 2>/dev/null || true
        echo -e "${GREEN}✅ Cleanup complete${NC}"
        ;;
    *)
        echo -e "${YELLOW}Usage: $0 {build|run|stop|restart|logs|status|shell|clean}${NC}"
        echo ""
        echo "Commands:"
        echo "  build   - Build the Docker image only"
        echo "  run     - Build and run the container (default)"
        echo "  stop    - Stop the running container"
        echo "  restart - Restart the container"
        echo "  logs    - Show container logs"
        echo "  status  - Show container status"
        echo "  shell   - Open bash shell in container"
        echo "  clean   - Stop container and remove image"
        exit 1
        ;;
esac
